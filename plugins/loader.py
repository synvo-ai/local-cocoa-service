"""
Plugin Loader
Dynamically discovers and loads plugin routers with dependency isolation
"""

from __future__ import annotations

import json
import logging
import sys
import importlib
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from core.config import settings

if TYPE_CHECKING:
    from fastapi import FastAPI, APIRouter

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Plugin manifest information"""
    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    category: str = "custom"
    
    # Backend configuration
    backend_entrypoint: str = "router"
    router_module: Optional[str] = None
    db_init: Optional[str] = None
    db_migrate: Optional[str] = None
    requirements: Optional[str] = None
    standalone: bool = False  # If True, plugin runs as standalone server (e.g., MCP)
    
    # Runtime state
    path: Path = field(default_factory=Path)
    loaded: bool = False
    error: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], plugin_path: Path) -> "PluginMetadata":
        """Create PluginMetadata from manifest dict"""
        backend_config = data.get("backend", {})
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            category=data.get("category", "custom"),
            backend_entrypoint=backend_config.get("entrypoint", "router"),
            router_module=backend_config.get("routerModule"),
            db_init=backend_config.get("dbInit"),
            db_migrate=backend_config.get("dbMigrate"),
            requirements=backend_config.get("requirements"),
            standalone=backend_config.get("standalone", False),
            path=plugin_path,
        )


class PluginLoader:
    """
    Manages plugin discovery and loading for the Python backend.
    
    Key features:
    - Dynamic router registration with unique API prefix per plugin
    - Dependency isolation via sys.path manipulation
    - Database table name prefixing
    
    All plugins (including built-in ones like Activity, Mail, Notes) are stored
    in the same plugins/ directory for simplicity.
    """
    
    MANIFEST_NAME = "plugin.json"
    
    def __init__(self, sys_plugs_dir: Path, user_plugs_dir: Optional[Path] = None):
        """
        Initialize the plugin loader.
        
        Args:
            sys_plugs_dir: System plugins root directory of the application
            user_plugs_dir: User plugins (downloadable) root directory of the application
        """
        self.plugins_path_list = [sys_plugs_dir]
        if user_plugs_dir:
            self.plugins_path_list.append(user_plugs_dir)
        
        self.plugins: Dict[str, PluginMetadata] = {}
        self._original_sys_path: List[str] = []
        
        logger.info(f"PluginLoader initialized. Plugins directory: {self.plugins_path_list}")
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """
        Discover all available plugins in the plugins/ directory.
        
        Returns:
            List of discovered plugin metadata
        """
        discovered = []
        
        for plugins_path in self.plugins_path_list:
            if not plugins_path.exists():
                logger.warning(f"Plugins directory not found: {plugins_path}")
                continue
            
            for plugin_dir in plugins_path.iterdir():
                if plugin_dir.is_dir():
                    metadata = self._load_plugin_metadata(plugin_dir)
                    if metadata:
                        discovered.append(metadata)
                        self.plugins[metadata.id] = metadata
            
        
        logger.info(f"Discovered {len(discovered)} plugins: {[p.id for p in discovered]}")
        return discovered
    
    def _load_plugin_metadata(self, plugin_dir: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from manifest file"""
        manifest_path = plugin_dir / self.MANIFEST_NAME
        
        if not manifest_path.exists():
            logger.warning(f"No manifest found in {plugin_dir}")
            return None
        
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            
            # Validate required fields
            if not all(k in manifest_data for k in ["id", "name", "version"]):
                logger.warning(f"Invalid manifest in {plugin_dir}: missing required fields")
                return None
            
            metadata = PluginMetadata.from_dict(manifest_data, plugin_dir)
            logger.debug(f"Loaded plugin metadata: {metadata.name} ({metadata.id})")
            return metadata
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse manifest in {plugin_dir}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading plugin metadata from {plugin_dir}: {e}")
            return None
    
    def register_all_routers(self, app: "FastAPI") -> int:
        """
        Register all discovered plugin routers with the FastAPI app.
        
        Each plugin gets a unique prefix: /plugins/{plugin_id}/
        
        Args:
            app: FastAPI application instance
            
        Returns:
            Number of successfully registered plugins
        """
        registered = 0
        
        for plugin_id, metadata in self.plugins.items():
            if not metadata.backend_entrypoint:
                continue
            
            # Skip standalone plugins (e.g., MCP server) - they run as separate processes
            if metadata.standalone:
                logger.debug(f"Skipping standalone plugin: {metadata.id}")
                continue
                
            try:
                router = self._load_plugin_router(metadata)
                if router:
                    # Register with unique prefix
                    prefix = f"/plugins/{plugin_id}"
                    app.include_router(router, prefix=prefix, tags=[f"plugin:{metadata.name}"])
                    metadata.loaded = True
                    registered += 1
                    logger.info(f"Registered plugin router: {metadata.name} at {prefix}")
            except Exception as e:
                metadata.error = str(e)
                logger.error(f"Failed to register plugin {plugin_id}: {e}")
                try:
                    with open("plugin_errors.log", "a") as f:
                        import traceback
                        f.write(f"Failed to register plugin {plugin_id}: {e}\n")
                        traceback.print_exc(file=f)
                except:
                    pass
        
        return registered
    
    def _load_plugin_router(self, metadata: PluginMetadata) -> Optional["APIRouter"]:
        """
        Load a plugin's FastAPI router with dependency isolation.
        
        Uses sys.path manipulation to ensure the plugin can find its dependencies
        without affecting other plugins.
        
        Always loads router.py directly to avoid circular import issues.
        """
        backend_path = metadata.path / "backend"
        
        if not backend_path.exists():
            logger.warning(f"Plugin {metadata.id} has no backend directory")
            return None
        
        # Save original sys.path
        self._original_sys_path = sys.path.copy()
        
        try:
            # Add plugin path to sys.path for package imports
            # This allows the plugin to import its own modules as a package
            sys.path.insert(0, str(metadata.path))
            sys.path.insert(0, str(backend_path))
            
            # Also add plugin's venv if it exists
            venv_path = backend_path / "venv"
            if venv_path.exists():
                site_packages = self._find_site_packages(venv_path)
                if site_packages:
                    sys.path.insert(0, str(site_packages))
            
            # Get unique package name for this plugin to ensure consistent module instances
            package_name = self._get_plugin_package_name(metadata.id)
            
            # Ensure the package module exists in sys.modules
            # This allows relative imports like 'from .service import ...' to work correctly
            if package_name not in sys.modules:
                pkg = type(sys)(package_name)
                pkg.__path__ = [str(backend_path)]
                pkg.__package__ = package_name
                sys.modules[package_name] = pkg
            
            # Load the router module as a submodule of the plugin package (e.g., plugin_mail.router)
            # CRITICAL: Use dot notation so relative imports 'from .service' point to the same package
            module_name_to_load = metadata.backend_entrypoint
            module_name = f"{package_name}.{module_name_to_load}"
            router_path = backend_path / f"{module_name_to_load}.py"
            
            if not router_path.exists():
                logger.error(f"Plugin router module not found: {router_path}")
                return None
            
            # Register the module but don't use 'backend' alias as it conflicts between plugins
            spec = importlib.util.spec_from_file_location(
                module_name,
                router_path,
                submodule_search_locations=[str(backend_path)]
            )
            if not spec or not spec.loader:
                logger.error(f"Failed to create module spec for {router_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            # CRITICAL: Set the package attribute so relative imports work and are consistent
            module.__package__ = package_name
            
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Get the router from the module
            router = getattr(module, "router", None)
            if router is None:
                logger.error(f"Plugin {metadata.id} module has no 'router' attribute")
                return None
            
            return router
        except Exception:
            logger.error(f"Failed to load plugin router for {metadata.id}: ")
            exc_type, exc_msg, tb = sys.exc_info()
            logger.error(f"Exception location: {tb.tb_frame.f_code.co_filename}，line {tb.tb_lineno}, | {exc_type.__name__}: {exc_msg}")
            return None
            
        finally:
            # Restore original sys.path
            sys.path = self._original_sys_path
    
    def _find_site_packages(self, venv_path: Path) -> Optional[Path]:
        """Find site-packages directory in a virtualenv"""
        # Try common locations
        candidates = [
            venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
            venv_path / "Lib" / "site-packages",  # Windows
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        return None

    
    def _get_plugin_package_name(self, plugin_id: str) -> str:
        """Get a consistent sanitized package name for a plugin (no dots or dashes)"""
        sanitized = plugin_id.replace("-", "_").replace(".", "_")
        return f"plugin_{sanitized}"

    def get_plugin(self, plugin_id: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by ID"""
        return self.plugins.get(plugin_id)
    
    def get_plugin_api_base(self, plugin_id: str) -> Optional[str]:
        """
        Get the API base path for a plugin.
        Useful for inter-plugin communication.
        """
        if plugin_id in self.plugins:
            return f"/plugins/{plugin_id}"
        return None
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins with their status"""
        return [
            {
                "id": m.id,
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "author": m.author,
                "category": m.category,
                "loaded": m.loaded,
                "error": m.error,
            }
            for m in self.plugins.values()
        ]
    
    def unload_plugin(self, app: "FastAPI", plugin_id: str) -> bool:
        """
        Unload a plugin's router from the FastAPI app.
        
        This enables hot-reload by removing routes before re-adding them.
        
        Args:
            app: FastAPI application instance
            plugin_id: Plugin ID to unload
            
        Returns:
            True if plugin was unloaded, False if not found
        """
        if plugin_id not in self.plugins:
            logger.warning(f"Plugin not found for unload: {plugin_id}")
            return False
        
        metadata = self.plugins[plugin_id]
        
        # Remove routes with matching prefix
        prefix = f"/plugins/{plugin_id}"
        routes_to_remove = []
        
        for i, route in enumerate(app.router.routes):
            # Check if route path starts with our plugin prefix
            route_path = getattr(route, "path", "")
            if route_path.startswith(prefix):
                routes_to_remove.append(i)
        
        # Remove routes in reverse order to avoid index shifting
        for i in sorted(routes_to_remove, reverse=True):
            del app.router.routes[i]
        
        metadata.loaded = False
        logger.info(f"Unloaded plugin: {plugin_id} (removed {len(routes_to_remove)} routes)")
        return True
    
    def reload_plugin(self, app: "FastAPI", plugin_id: str) -> bool:
        """
        Hot-reload a plugin by unloading and re-loading it.
        
        Args:
            app: FastAPI application instance
            plugin_id: Plugin ID to reload
            
        Returns:
            True if plugin was reloaded successfully
        """
        if plugin_id not in self.plugins:
            logger.warning(f"Plugin not found for reload: {plugin_id}")
            return False
        
        # Unload existing routes
        self.unload_plugin(app, plugin_id)
        
        # Re-read manifest in case it changed
        metadata = self.plugins[plugin_id]
        manifest_path = metadata.path / self.MANIFEST_NAME
        
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                
                # Update metadata
                new_metadata = PluginMetadata.from_dict(manifest_data, metadata.path)
                self.plugins[plugin_id] = new_metadata
                metadata = new_metadata
            except Exception as e:
                logger.error(f"Failed to re-read manifest for {plugin_id}: {e}")
        
        # Clear any cached modules for this plugin
        modules_to_remove = [
            name for name in sys.modules
            if name.startswith(f"plugin_{plugin_id}")
        ]
        for module_name in modules_to_remove:
            del sys.modules[module_name]
        
        # Re-load the router
        try:
            router = self._load_plugin_router(metadata)
            if router:
                prefix = f"/plugins/{plugin_id}"
                app.include_router(router, prefix=prefix, tags=[f"plugin:{metadata.name}"])
                metadata.loaded = True
                logger.info(f"Reloaded plugin: {plugin_id}")
                return True
        except Exception as e:
            metadata.error = str(e)
            logger.error(f"Failed to reload plugin {plugin_id}: {e}")
        
        return False
    
    def refresh_plugins(self, app: "FastAPI") -> int:
        """
        Refresh all plugins by re-discovering and re-loading.
        
        Args:
            app: FastAPI application instance
            
        Returns:
            Number of plugins reloaded
        """
        logger.info("Refreshing all plugins...")
        
        # Unload all currently loaded plugins
        for plugin_id, metadata in self.plugins.items():
            if metadata.loaded:
                self.unload_plugin(app, plugin_id)
        
        # Clear plugin registry
        self.plugins.clear()
        
        # Re-discover
        self.discover_plugins()
        
        # Re-register all
        registered = self.register_all_routers(app)
        
        # Reload services
        # Note: In a real app, you might want to pass storage/indexer here
        # For simplicity, we assume they are already available in core.context
        from core.context import indexer
        self.register_all_services(indexer)
        
        return registered

    def register_all_services(self, indexer: Any) -> int:
        """
        Discover and register services from each plugin.
        Expected entrypoint: backend/service.py with a class named {PluginID}Service
        """
        from core.context import register_service
        registered = 0
        
        for plugin_id, metadata in self.plugins.items():
            backend_path = metadata.path / "backend"
            service_path = backend_path / "service.py"
            
            if not service_path.exists():
                continue
            
            # Save original sys.path
            current_sys_path = sys.path.copy()

            try:
                # Use consistent package naming logic
                package_name = self._get_plugin_package_name(plugin_id)
                
                # Ensure the package exists in sys.modules
                if package_name not in sys.modules:
                    pkg = type(sys)(package_name)
                    pkg.__path__ = [str(backend_path)]
                    pkg.__package__ = package_name
                    sys.modules[package_name] = pkg
                
                # Load the service module as a submodule (e.g., plugin_mail.service)
                module_name = f"{package_name}.service"
                
                # If already loaded (e.g. by router), just use it
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                else:
                    spec = importlib.util.spec_from_file_location(module_name, service_path)
                    if not spec or not spec.loader:
                        continue
                    
                    module = importlib.util.module_from_spec(spec)
                    # CRITICAL: Set the package attribute so relative imports work
                    module.__package__ = package_name
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                
                # Check for factory function
                get_service_func = getattr(module, "init_plugin_service", None)
                
                if get_service_func and callable(get_service_func):
                    logger.info(f"Initializing service for plugin {plugin_id}")
                    try:
                        # Try factory with arguments
                        service_instance = get_service_func(indexer, plugin_id=plugin_id)
                        if service_instance:
                            register_service(plugin_id, service_instance)
                            registered += 1
                        else:
                             # Some services might return None if disabled/failed
                             logger.warning(f"Plugin {plugin_id} factory returned None")

                    except TypeError:
                        # Fallback for factories that might not accept arguments (though they should)
                        # or other TypeError. Try without arguments?
                        logger.warning(f"Plugin {plugin_id} factory failed with arguments, trying without args.")
                        try:
                            service_instance = get_service_func() # type: ignore
                            if service_instance:
                                register_service(plugin_id, service_instance)
                                registered += 1
                        except Exception as e2:
                            logger.error(f"Plugin {plugin_id} factory failed: {e2}")
                else:
                    logger.debug(f"No get_plugin_service factory found for plugin {plugin_id}")

            except Exception as e:
                logger.error(f"Failed to load service for plugin {plugin_id}: {e}")
            finally:
                # Restore sys.path
                sys.path = current_sys_path
                
        return registered
    
    def get_db_table_prefix(self, plugin_id: str) -> str:
        """
        Get database table prefix for a plugin.
        All plugin tables should be prefixed with this to avoid conflicts.
        
        Format: plugin_{sanitized_id}_
        """
        # Sanitize plugin ID for table naming
        sanitized = plugin_id.replace(".", "_").replace("-", "_").lower()
        return f"plugin_{sanitized}_"


# Global singleton
_plugin_loader: Optional[PluginLoader] = None


def init_plugin_loader(sys_plugins_dir: Path, user_plugins_dir: Optional[Path] = None) -> PluginLoader:
    """Initialize the global plugin loader with system and optional user plugin directories"""
    global _plugin_loader
    _plugin_loader = PluginLoader(sys_plugins_dir, user_plugins_dir)
    return _plugin_loader


def get_plugin_loader() -> Optional[PluginLoader]:
    """Get the global plugin loader instance"""
    return _plugin_loader

def init_all_plugins(app: "FastAPI"):
    try:
        # Initialize with both system and user plugin directories
        plugin_loader = init_plugin_loader(
            settings.paths.plugins_root,
            settings.paths.user_plugins_root
        )
        plugin_loader.discover_plugins()

        registered_count = plugin_loader.register_all_routers(app)
        
        # Load plugin services
        from core.context import indexer
        services_count = plugin_loader.register_all_services(indexer)
        
        logger.info(f"Plugin system initialized: {registered_count} routers, {services_count} services loaded")
    except Exception as e:
        logger.error(f"Failed to initialize plugin system: {e}")

