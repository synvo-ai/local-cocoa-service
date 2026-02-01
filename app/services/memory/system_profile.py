"""
Mac System Profile → Semantic Profile Inference

This module collects Mac system information and uses LLM to infer a semantic user profile.

Flow:
1. Collect raw system data (apps, dev tools, settings, username)
2. Use LLM to infer semantic profile (personality, interests, skills, etc.)
3. Return ProfileRecord-compatible data

The inferred profile includes:
- personality: 性格特征
- interests: 兴趣爱好
- hard_skills / soft_skills: 技能
- working_habit_preference: 工作习惯偏好
- user_goal: 用户目标
- motivation_system: 动机系统
- value_system: 价值观
"""

import os
import subprocess
import logging
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from core.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Profile Persistence
# ═══════════════════════════════════════════════════════════════════════════════

def _get_profile_path(user_id: str) -> Path:
    """Get the file path for storing a user's basic profile."""
    profile_dir = settings.base_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir / f"basic_profile_{user_id}.json"


def save_basic_profile(user_id: str, profile_data: Dict[str, Any]) -> None:
    """Save basic profile to disk."""
    path = _get_profile_path(user_id)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, ensure_ascii=False, indent=2)
        logger.info("Saved basic profile for user %s to %s", user_id, path)
    except Exception as e:
        logger.error("Failed to save basic profile: %s", e)


def load_basic_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Load basic profile from disk. Returns None if not found."""
    path = _get_profile_path(user_id)
    if not path.exists():
        logger.info("No cached profile found for user %s", user_id)
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("Loaded cached profile for user %s from %s", user_id, path)
        return data
    except Exception as e:
        logger.error("Failed to load basic profile: %s", e)
        return None


def _save_profile_data(user_id: str, raw_data, topics: List[Dict], scanned_at: str) -> None:
    """Internal helper to save profile data after generation."""
    profile_data = {
        "user_id": user_id,
        "user_name": raw_data.username,
        "raw_system_data": {
            "username": raw_data.username,
            "computer_name": raw_data.computer_name,
            "shell": raw_data.shell,
            "language": raw_data.language,
            "region": raw_data.region,
            "timezone": raw_data.timezone,
            "appearance": raw_data.appearance,
            "installed_apps": raw_data.installed_apps,
            "dev_tools": raw_data.dev_tools,
        },
        "topics": topics,
        "scanned_at": scanned_at,
    }
    save_basic_profile(user_id, profile_data)


# ═══════════════════════════════════════════════════════════════════════════════
# Raw System Data Collection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RawSystemData:
    """Raw system data collected from Mac."""
    username: str = ""
    computer_name: str = ""
    home_directory: str = ""
    shell: str = ""
    language: str = ""
    region: str = ""
    timezone: str = ""
    appearance: str = ""  # "light" or "dark"
    installed_apps: List[str] = field(default_factory=list)
    dev_tools: List[Dict[str, str]] = field(default_factory=list)  # [{name, version}]


def _run_command(cmd: list[str], default: str = "") -> str:
    """Run a shell command and return output, or default on error."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else default
    except Exception as e:
        logger.debug("Command %s failed: %s", cmd, e)
        return default


def _get_defaults_value(domain: str, key: str, default: str = "") -> str:
    """Read a value from macOS defaults."""
    return _run_command(["defaults", "read", domain, key], default)


def collect_raw_system_data() -> RawSystemData:
    """Collect raw system information from Mac."""
    logger.info("Starting raw system data collection...")

    # Basic info
    username = os.environ.get("USER", "")
    home = os.environ.get("HOME", "")
    shell = os.environ.get("SHELL", "")
    logger.debug("Basic info: user=%s, home=%s, shell=%s", username, home, shell)

    computer_name = _run_command(["scutil", "--get", "ComputerName"])
    if not computer_name:
        computer_name = _run_command(["hostname", "-s"])

    # Language
    languages = _get_defaults_value("-g", "AppleLanguages")
    language = ""
    if languages:
        for line in languages.split("\n"):
            line = line.strip().strip('(",)')
            if line and not line.startswith("("):
                language = line
                break

    # Region
    region = _get_defaults_value("-g", "AppleLocale")

    # Timezone - use /etc/localtime symlink (doesn't require admin)
    try:
        localtime_path = Path("/etc/localtime")
        if localtime_path.is_symlink():
            # Read the symlink target directly (don't resolve)
            tz_link = os.readlink("/etc/localtime")
            # Extract timezone from path like /var/db/timezone/zoneinfo/Asia/Shanghai
            if "zoneinfo/" in tz_link:
                timezone_val = tz_link.split("zoneinfo/")[-1]
            elif "/zoneinfo" in tz_link:
                # Handle paths like /usr/share/zoneinfo/Asia/Shanghai
                timezone_val = tz_link.split("/zoneinfo")[-1].lstrip("/")
            else:
                # Fallback to Python time module
                import time
                timezone_val = time.tzname[0] if time.tzname else ""
        else:
            import time
            timezone_val = time.tzname[0] if time.tzname else ""
        logger.debug("Detected timezone: %s", timezone_val)
    except Exception as e:
        logger.debug("Failed to get timezone: %s", e)
        timezone_val = ""

    # Appearance
    appearance_raw = _get_defaults_value("-g", "AppleInterfaceStyle")
    appearance = "dark" if appearance_raw.lower() == "dark" else "light"

    # Installed apps
    installed_apps = []
    app_dirs = [Path("/Applications"), Path.home() / "Applications"]
    logger.info("Scanning for apps in: %s", app_dirs)
    for app_dir in app_dirs:
        if not app_dir.exists():
            logger.debug("App directory does not exist: %s", app_dir)
            continue
        try:
            for item in app_dir.iterdir():
                if item.suffix == ".app":
                    installed_apps.append(item.stem)
        except PermissionError as e:
            logger.warning("Permission denied scanning %s: %s", app_dir, e)
        except Exception as e:
            logger.warning("Error scanning %s: %s", app_dir, e)
    installed_apps.sort(key=lambda x: x.lower())
    logger.info("Found %d installed apps", len(installed_apps))

    # Dev tools - search in common PATH locations explicitly
    # Backend process might not have the same PATH as user's shell
    dev_tools = []
    common_paths = [
        "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin",
        str(Path.home() / ".cargo/bin"),  # Rust
        str(Path.home() / "go/bin"),      # Go
        str(Path.home() / ".local/bin"),  # pipx, etc.
        "/Applications/Docker.app/Contents/Resources/bin",  # Docker
    ]

    # Build extended PATH for subprocess
    extended_path = ":".join(common_paths) + ":" + os.environ.get("PATH", "")
    env_with_path = {**os.environ, "PATH": extended_path}

    tools_to_check = [
        ("git", ["git", "--version"]),
        ("node", ["node", "--version"]),
        ("python3", ["python3", "--version"]),
        ("docker", ["docker", "--version"]),
        ("java", ["java", "-version"]),
        ("go", ["go", "version"]),
        ("rust", ["rustc", "--version"]),
        ("swift", ["swift", "--version"]),
        ("ruby", ["ruby", "--version"]),
        ("brew", ["brew", "--version"]),
        ("npm", ["npm", "--version"]),
        ("yarn", ["yarn", "--version"]),
        ("pnpm", ["pnpm", "--version"]),
        ("cargo", ["cargo", "--version"]),
        ("pip3", ["pip3", "--version"]),
    ]

    for name, version_cmd in tools_to_check:
        # Try to find the tool in extended PATH
        try:
            which_result = subprocess.run(
                ["which", name],
                capture_output=True, text=True, timeout=5,
                env=env_with_path
            )
            if which_result.returncode != 0:
                continue
        except Exception:
            continue

        version = ""
        try:
            result = subprocess.run(
                version_cmd,
                capture_output=True, text=True, timeout=5,
                env=env_with_path
            )
            output = result.stdout.strip() or result.stderr.strip()
            if output:
                version = output.split("\n")[0]
        except Exception:
            pass
        dev_tools.append({"name": name, "version": version})
        logger.debug("Found dev tool: %s (%s)", name, version)

    logger.info(
        "System data collection complete: user=%s, computer=%s, %d apps, %d dev tools",
        username, computer_name, len(installed_apps), len(dev_tools)
    )

    return RawSystemData(
        username=username,
        computer_name=computer_name,
        home_directory=home,
        shell=shell,
        language=language,
        region=region,
        timezone=timezone_val,
        appearance=appearance,
        installed_apps=installed_apps,
        dev_tools=dev_tools,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-based Profile Inference
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROFILE_INFERENCE_PROMPT = """You are a user profiling expert. Based on the following Mac system information, infer a comprehensive semantic profile for this user.

## System Information

**User:** {username} ({computer_name})
**Shell:** {shell}
**Language:** {language}
**Region:** {region}
**Timezone:** {timezone}
**Appearance Mode:** {appearance}

**Installed Applications ({app_count} total):**
{apps_list}

**Development Tools:**
{dev_tools_list}

## Task

Based on this system information, infer the user's semantic profile across multiple dimensions. Consider:
- What applications reveal about their profession, interests, and lifestyle
- Development tools indicate technical skills and programming languages
- System preferences suggest work habits, culture, and personal style
- App combinations reveal potential roles (designer, developer, content creator, etc.)
- Regional settings suggest nationality, language preferences

**Important:**
- Be creative and insightful - don't just list the obvious
- Consider combinations of apps that reveal deeper patterns
- Infer personality traits from tool choices and preferences
- This is for a general user, not necessarily a programmer
- For each inferred value, provide confidence level and evidence

## Output Format

Return ONLY a JSON object with hierarchical topics. Each subtopic should have:
- "value": the inferred value (string, array, or null if unknown)
- "confidence": "high" / "medium" / "low" (based on evidence strength)
- "evidence": brief explanation of what data supported this inference

```json
{{
    "user_name": "inferred display name",
    "topics": [
        {{
            "topic_id": "basic_info",
            "topic_name": "Basic Information",
            "subtopics": [
                {{"name": "name", "value": "...", "confidence": "high", "evidence": "from username"}},
                {{"name": "language_spoken", "value": ["English", "Chinese"], "confidence": "high", "evidence": "system language en-US, region CN"}},
                {{"name": "nationality", "value": "...", "confidence": "medium", "evidence": "..."}}
            ]
        }},
        {{
            "topic_id": "work",
            "topic_name": "Work & Career",
            "subtopics": [
                {{"name": "working_industry", "value": "...", "confidence": "...", "evidence": "..."}},
                {{"name": "title", "value": "...", "confidence": "...", "evidence": "..."}},
                {{"name": "work_skills", "value": [...], "confidence": "...", "evidence": "..."}}
            ]
        }},
        {{
            "topic_id": "interest",
            "topic_name": "Interests & Hobbies",
            "subtopics": [
                {{"name": "music", "value": true, "confidence": "high", "evidence": "Logic Pro, Spotify installed"}},
                {{"name": "movies", "value": true, "confidence": "medium", "evidence": "..."}},
                {{"name": "hobbies", "value": [...], "confidence": "...", "evidence": "..."}}
            ]
        }},
        {{
            "topic_id": "psychological",
            "topic_name": "Psychological Profile",
            "subtopics": [
                {{"name": "personality", "value": ["creative", "detail-oriented"], "confidence": "medium", "evidence": "..."}},
                {{"name": "values", "value": [...], "confidence": "...", "evidence": "..."}},
                {{"name": "motivations", "value": [...], "confidence": "...", "evidence": "..."}},
                {{"name": "goals", "value": [...], "confidence": "...", "evidence": "..."}}
            ]
        }},
        {{
            "topic_id": "behavioral",
            "topic_name": "Behavioral Patterns",
            "subtopics": [
                {{"name": "working_habit", "value": [...], "confidence": "...", "evidence": "..."}},
                {{"name": "decision_making_style", "value": "...", "confidence": "...", "evidence": "..."}},
                {{"name": "communication_style", "value": "...", "confidence": "...", "evidence": "..."}}
            ]
        }},
        {{
            "topic_id": "technical",
            "topic_name": "Technical Profile",
            "subtopics": [
                {{"name": "hard_skills", "value": [{{"skill": "...", "level": "advanced"}}], "confidence": "high", "evidence": "..."}},
                {{"name": "soft_skills", "value": [...], "confidence": "...", "evidence": "..."}},
                {{"name": "tools_used", "value": [...], "confidence": "high", "evidence": "..."}},
                {{"name": "inferred_roles", "value": ["Developer", "Designer"], "confidence": "...", "evidence": "..."}}
            ]
        }}
    ]
}}
```

Only include topics where you can make reasonable inferences. Skip topics with no evidence.
Do not include any text outside the JSON block.
"""


@dataclass
class ProfileSubtopicData:
    """A single subtopic with inferred value."""
    name: str
    value: Any = None
    confidence: Optional[str] = None  # high, medium, low
    evidence: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ProfileTopicData:
    """A topic containing multiple subtopics."""
    topic_id: str
    topic_name: str
    subtopics: List[ProfileSubtopicData] = field(default_factory=list)
    icon: Optional[str] = None


@dataclass
class InferredProfile:
    """LLM-inferred semantic profile with hierarchical structure."""
    user_id: str
    user_name: Optional[str] = None

    # Hierarchical topics
    topics: List[ProfileTopicData] = field(default_factory=list)

    # Legacy flat fields (for backward compatibility)
    personality: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    hard_skills: List[Dict[str, str]] = field(default_factory=list)
    soft_skills: List[Dict[str, str]] = field(default_factory=list)
    working_habit_preference: List[str] = field(default_factory=list)
    user_goal: List[str] = field(default_factory=list)
    motivation_system: List[str] = field(default_factory=list)
    value_system: List[str] = field(default_factory=list)
    inferred_roles: List[str] = field(default_factory=list)

    # Raw system data for reference
    raw_system_data: Optional[Dict[str, Any]] = None
    scanned_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert ProfileTopicData to dict
        result["topics"] = [
            {
                "topic_id": t.topic_id,
                "topic_name": t.topic_name,
                "icon": t.icon,
                "subtopics": [
                    {
                        "name": s.name,
                        "value": s.value,
                        "confidence": s.confidence,
                        "evidence": s.evidence,
                        "description": s.description,
                    }
                    for s in t.subtopics
                ]
            }
            for t in self.topics
        ]
        return result


def _parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response JSON."""
    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try direct JSON parse
    return json.loads(response)


async def infer_profile_with_llm(
    raw_data: RawSystemData,
    user_id: str = "default_user",
    llm_client = None,
) -> InferredProfile:
    """
    Use LLM to infer semantic profile from raw system data.

    Args:
        raw_data: Collected Mac system information
        user_id: User ID for the profile
        llm_client: LLM client instance (from services.llm.client.LlmClient)

    Returns:
        InferredProfile with LLM-inferred semantic data
    """
    # Build prompt
    apps_list = ", ".join(raw_data.installed_apps[:50])  # Limit to 50 apps
    if len(raw_data.installed_apps) > 50:
        apps_list += f" ... and {len(raw_data.installed_apps) - 50} more"

    dev_tools_list = "\n".join([
        f"- {t['name']}: {t['version']}" for t in raw_data.dev_tools
    ]) if raw_data.dev_tools else "None detected"

    prompt = SYSTEM_PROFILE_INFERENCE_PROMPT.format(
        username=raw_data.username,
        computer_name=raw_data.computer_name,
        shell=raw_data.shell,
        language=raw_data.language or "Not set",
        region=raw_data.region or "Not set",
        timezone=raw_data.timezone or "Not set",
        appearance=raw_data.appearance,
        app_count=len(raw_data.installed_apps),
        apps_list=apps_list,
        dev_tools_list=dev_tools_list,
    )

    # Default profile if LLM fails - include full raw_system_data
    default_profile = InferredProfile(
        user_id=user_id,
        user_name=raw_data.username,
        raw_system_data={
            "username": raw_data.username,
            "computer_name": raw_data.computer_name,
            "shell": raw_data.shell,
            "language": raw_data.language,
            "region": raw_data.region,
            "timezone": raw_data.timezone,
            "appearance": raw_data.appearance,
            "installed_apps": raw_data.installed_apps,
            "dev_tools": raw_data.dev_tools,
        },
        scanned_at=datetime.now(timezone.utc).isoformat(),
    )

    if llm_client is None:
        logger.warning("No LLM client provided, returning default profile with %d apps, %d dev tools",
                       len(raw_data.installed_apps), len(raw_data.dev_tools))
        return default_profile

    try:
        logger.info("Inferring profile with LLM for user: %s (apps=%d, tools=%d)",
                    user_id, len(raw_data.installed_apps), len(raw_data.dev_tools))

        # Call LLM (method is 'complete', not 'generate')
        response = await llm_client.complete(prompt, temperature=0.7, max_tokens=2048)

        if not response:
            logger.warning("Empty LLM response")
            return default_profile

        # Parse response
        parsed = _parse_llm_response(response)

        # Parse hierarchical topics
        topics: List[ProfileTopicData] = []
        for topic_data in parsed.get("topics", []):
            subtopics = [
                ProfileSubtopicData(
                    name=s.get("name", ""),
                    value=s.get("value"),
                    confidence=s.get("confidence"),
                    evidence=s.get("evidence"),
                    description=s.get("description"),
                )
                for s in topic_data.get("subtopics", [])
            ]
            topics.append(ProfileTopicData(
                topic_id=topic_data.get("topic_id", ""),
                topic_name=topic_data.get("topic_name", ""),
                subtopics=subtopics,
                icon=topic_data.get("icon"),
            ))

        # Extract legacy flat fields from topics for backward compatibility
        personality = []
        interests = []
        hard_skills = []
        soft_skills = []
        working_habit_preference = []
        user_goal = []
        motivation_system = []
        value_system = []
        inferred_roles = []

        for topic in topics:
            for sub in topic.subtopics:
                val = sub.value
                if val is None:
                    continue
                name_lower = sub.name.lower()

                if name_lower == "personality" and isinstance(val, list):
                    personality.extend(val)
                elif name_lower in ("interests", "hobbies") and isinstance(val, list):
                    interests.extend(val)
                elif name_lower == "hard_skills" and isinstance(val, list):
                    hard_skills.extend(val if isinstance(val[0], dict) else [{"name": v, "level": None} for v in val])
                elif name_lower == "soft_skills" and isinstance(val, list):
                    soft_skills.extend(val if isinstance(val[0], dict) else [{"name": v, "level": None} for v in val])
                elif name_lower in ("working_habit", "working_habit_preference") and isinstance(val, list):
                    working_habit_preference.extend(val)
                elif name_lower in ("goals", "user_goal") and isinstance(val, list):
                    user_goal.extend(val)
                elif name_lower in ("motivations", "motivation_system") and isinstance(val, list):
                    motivation_system.extend(val)
                elif name_lower in ("values", "value_system") and isinstance(val, list):
                    value_system.extend(val)
                elif name_lower == "inferred_roles" and isinstance(val, list):
                    inferred_roles.extend(val)

        profile = InferredProfile(
            user_id=user_id,
            user_name=parsed.get("user_name", raw_data.username),
            topics=topics,
            # Legacy flat fields
            personality=personality,
            interests=interests,
            hard_skills=hard_skills,
            soft_skills=soft_skills,
            working_habit_preference=working_habit_preference,
            user_goal=user_goal,
            motivation_system=motivation_system,
            value_system=value_system,
            inferred_roles=inferred_roles,
            raw_system_data={
                "username": raw_data.username,
                "computer_name": raw_data.computer_name,
                "shell": raw_data.shell,
                "language": raw_data.language,
                "region": raw_data.region,
                "timezone": raw_data.timezone,
                "appearance": raw_data.appearance,
                "installed_apps": raw_data.installed_apps,
                "dev_tools": raw_data.dev_tools,
            },
            scanned_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "Profile inferred: %d topics, %d personality traits, %d interests, %d roles",
            len(profile.topics),
            len(profile.personality),
            len(profile.interests),
            len(profile.inferred_roles),
        )

        return profile

    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON: %s", e)
        logger.debug("Raw LLM response that failed to parse: %s", response[:500] if response else "None")
        return default_profile
    except AttributeError as e:
        logger.error("LLM client method error (check if using correct method): %s", e)
        return default_profile
    except Exception as e:
        logger.error("LLM inference failed: %s (type: %s)", e, type(e).__name__)
        import traceback
        logger.debug("Full traceback: %s", traceback.format_exc())
        return default_profile


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point (Sync wrapper for async function)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_system_profile_sync(user_id: str = "default_user") -> Dict[str, Any]:
    """
    Synchronous wrapper - collects raw data only (no LLM inference).
    Use the async version for full LLM inference.
    """
    raw_data = collect_raw_system_data()

    return {
        "user_id": user_id,
        "user_name": raw_data.username,
        "raw_system_data": {
            "username": raw_data.username,
            "computer_name": raw_data.computer_name,
            "shell": raw_data.shell,
            "language": raw_data.language,
            "region": raw_data.region,
            "timezone": raw_data.timezone,
            "appearance": raw_data.appearance,
            "installed_apps": raw_data.installed_apps,
            "dev_tools": raw_data.dev_tools,
        },
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }


async def collect_and_infer_profile(
    user_id: str = "default_user",
    llm_client = None,
) -> InferredProfile:
    """
    Main entry point: collect system data and infer profile with LLM.

    Args:
        user_id: User ID for the profile
        llm_client: LLM client for inference (from services.llm.client.LlmClient)

    Returns:
        InferredProfile with semantic profile data
    """
    logger.info("Collecting system data for user: %s", user_id)
    raw_data = collect_raw_system_data()

    logger.info(
        "System data collected: %d apps, %d dev tools",
        len(raw_data.installed_apps),
        len(raw_data.dev_tools),
    )

    profile = await infer_profile_with_llm(raw_data, user_id, llm_client)
    return profile


# ═══════════════════════════════════════════════════════════════════════════════
# Progressive/Streaming Profile Generation
# ═══════════════════════════════════════════════════════════════════════════════

# Topic definitions with their prompts
TOPIC_DEFINITIONS = {
    "basic_info": {
        "topic_name": "Basic Information",
        "icon": "user",
        "subtopics": ["name", "age", "gender", "nationality", "language_spoken", "location"],
        "prompt": """Based on the system info, infer basic user information:
- name: Display name (from username)
- language_spoken: Languages (from system language/region)
- nationality: Possible nationality (from region/timezone)
- location: Possible location (from timezone)

Only include fields you can reasonably infer."""
    },
    "technical": {
        "topic_name": "Technical Profile",
        "icon": "code",
        "subtopics": ["hard_skills", "soft_skills", "tools_used", "inferred_roles", "experience_level"],
        "prompt": """Based on the installed apps and dev tools, infer technical profile:
- hard_skills: Technical skills with levels (e.g., Python/advanced, React/intermediate)
- soft_skills: Soft skills like problem-solving, attention to detail
- tools_used: Main tools/technologies used
- inferred_roles: Likely job roles (Developer, Designer, etc.)
- experience_level: Overall experience level (junior/mid/senior)

Focus on what the apps and tools reveal about technical expertise."""
    },
    "work": {
        "topic_name": "Work & Career",
        "icon": "briefcase",
        "subtopics": ["working_industry", "title", "work_style", "work_responsibility"],
        "prompt": """Based on the apps, infer work-related information:
- working_industry: Likely industry (Tech, Finance, Creative, etc.)
- title: Possible job title
- work_style: Remote/office preference, work habits
- work_responsibility: Types of responsibilities based on tools used

Consider professional apps like Slack, Zoom, productivity tools, etc."""
    },
    "interest": {
        "topic_name": "Interests & Hobbies",
        "icon": "heart",
        "subtopics": ["music", "movies", "gaming", "reading", "sports", "creative_hobbies", "other_hobbies"],
        "prompt": """Based on entertainment and hobby apps, infer interests:
- music: Music interest (true/false), genres if possible
- movies: Movie/video interest
- gaming: Gaming interest, game types
- reading: Reading/learning interest
- sports: Sports/fitness interest
- creative_hobbies: Creative activities (art, photography, video editing, etc.)
- other_hobbies: Other notable interests

Look at apps like Spotify, Netflix, Steam, Kindle, etc."""
    },
    "psychological": {
        "topic_name": "Psychological Profile",
        "icon": "brain",
        "subtopics": ["personality", "values", "motivations", "goals", "decision_style"],
        "prompt": """Based on app choices and preferences, infer psychological traits:
- personality: Key personality traits (creative, analytical, organized, etc.)
- values: Core values (efficiency, quality, collaboration, etc.)
- motivations: What drives them (learning, achievement, creativity, etc.)
- goals: Possible life/career goals
- decision_style: How they likely make decisions

Consider what app combinations reveal about personality."""
    },
    "behavioral": {
        "topic_name": "Behavioral Patterns",
        "icon": "activity",
        "subtopics": ["working_habit", "communication_style", "learning_style", "organization_style"],
        "prompt": """Based on system preferences and apps, infer behavioral patterns:
- working_habit: Work habits (dark mode = night owl, productivity apps = organized, etc.)
- communication_style: Preferred communication (async/sync, tools used)
- learning_style: How they prefer to learn (videos, docs, hands-on)
- organization_style: How they organize (tools like Notion, notes apps, etc.)

Consider system preferences like dark mode, shell choice, etc."""
    },
}

# Order of topic generation (most useful first)
TOPIC_GENERATION_ORDER = [
    "basic_info",    # Instant (no LLM needed for some fields)
    "technical",     # Most important for many users
    "work",          # Career info
    "interest",      # Interests
    "psychological", # Personality
    "behavioral",    # Behavioral patterns
]


def extract_basic_info_instant(raw_data: RawSystemData, user_id: str) -> ProfileTopicData:
    """
    Extract basic info directly from system data (no LLM needed).
    This returns instantly.
    """
    subtopics = []

    # Name from username
    if raw_data.username:
        subtopics.append(ProfileSubtopicData(
            name="name",
            value=raw_data.username,
            confidence="high",
            evidence="From system username"
        ))

    # Language from system settings
    if raw_data.language:
        languages = [raw_data.language]
        if raw_data.region and raw_data.region != raw_data.language:
            # If region is different, might be bilingual
            region_lang = raw_data.region.split("_")[0] if "_" in raw_data.region else raw_data.region
            if region_lang not in raw_data.language:
                languages.append(region_lang)
        subtopics.append(ProfileSubtopicData(
            name="language_spoken",
            value=languages,
            confidence="high",
            evidence=f"System language: {raw_data.language}, region: {raw_data.region}"
        ))

    # Location from timezone
    if raw_data.timezone:
        location = raw_data.timezone.replace("_", " ")
        if "/" in location:
            location = location.split("/")[-1]
        subtopics.append(ProfileSubtopicData(
            name="location",
            value=location,
            confidence="medium",
            evidence=f"Timezone: {raw_data.timezone}"
        ))

    return ProfileTopicData(
        topic_id="basic_info",
        topic_name="Basic Information",
        icon="user",
        subtopics=subtopics
    )


SINGLE_TOPIC_PROMPT_TEMPLATE = """You are a user profiling expert. Based on the Mac system information below, infer ONLY the "{topic_name}" section.

## System Information

**User:** {username} ({computer_name})
**Shell:** {shell}
**Language:** {language}
**Region:** {region}
**Timezone:** {timezone}
**Appearance Mode:** {appearance}

**Installed Applications ({app_count} total):**
{apps_list}

**Development Tools:**
{dev_tools_list}

## Task

{topic_specific_prompt}

## Output Format

Return ONLY a JSON object with this structure:
```json
{{
    "topic_id": "{topic_id}",
    "topic_name": "{topic_name}",
    "subtopics": [
        {{"name": "field_name", "value": "inferred_value or [array]", "confidence": "high/medium/low", "evidence": "why you inferred this"}}
    ]
}}
```

Be concise. Only include fields with actual inferences. Return valid JSON only.
"""


async def generate_single_topic(
    topic_id: str,
    raw_data: RawSystemData,
    llm_client,
) -> Optional[ProfileTopicData]:
    """
    Generate a single topic using LLM.

    Args:
        topic_id: The topic to generate
        raw_data: System data
        llm_client: LLM client

    Returns:
        ProfileTopicData or None if failed
    """
    if topic_id not in TOPIC_DEFINITIONS:
        logger.warning("Unknown topic_id: %s", topic_id)
        return None

    topic_def = TOPIC_DEFINITIONS[topic_id]

    # Build apps and tools lists
    apps_list = ", ".join(raw_data.installed_apps[:50])
    if len(raw_data.installed_apps) > 50:
        apps_list += f" ... and {len(raw_data.installed_apps) - 50} more"

    dev_tools_list = "\n".join([
        f"- {t['name']}: {t['version']}" for t in raw_data.dev_tools
    ]) if raw_data.dev_tools else "None detected"

    prompt = SINGLE_TOPIC_PROMPT_TEMPLATE.format(
        topic_id=topic_id,
        topic_name=topic_def["topic_name"],
        topic_specific_prompt=topic_def["prompt"],
        username=raw_data.username,
        computer_name=raw_data.computer_name,
        shell=raw_data.shell,
        language=raw_data.language or "Not set",
        region=raw_data.region or "Not set",
        timezone=raw_data.timezone or "Not set",
        appearance=raw_data.appearance,
        app_count=len(raw_data.installed_apps),
        apps_list=apps_list,
        dev_tools_list=dev_tools_list,
    )

    try:
        logger.info("Generating topic: %s", topic_id)
        response = await llm_client.complete(prompt, temperature=0.7, max_tokens=800)

        if not response:
            logger.warning("Empty LLM response for topic: %s", topic_id)
            return None

        # Parse response
        parsed = _parse_llm_response(response)

        subtopics = [
            ProfileSubtopicData(
                name=s.get("name", ""),
                value=s.get("value"),
                confidence=s.get("confidence"),
                evidence=s.get("evidence"),
            )
            for s in parsed.get("subtopics", [])
        ]

        return ProfileTopicData(
            topic_id=topic_id,
            topic_name=topic_def["topic_name"],
            icon=topic_def.get("icon"),
            subtopics=subtopics
        )

    except Exception as e:
        logger.error("Failed to generate topic %s: %s", topic_id, e)
        return None


async def generate_profile_progressive(
    user_id: str = "default_user",
    llm_client = None,
):
    """
    Async generator that yields profile topics progressively.
    Saves the complete profile to disk after generation.

    Usage:
        async for event in generate_profile_progressive(user_id, llm_client):
            # event is dict with type and data
            print(event)

    Yields:
        dict with keys:
        - type: "init" | "topic" | "complete" | "error"
        - data: topic data or metadata
    """
    logger.info("Starting progressive profile generation for user: %s", user_id)

    # Step 1: Collect raw system data
    raw_data = collect_raw_system_data()

    # Track all generated topics for saving
    generated_topics = []
    scanned_at = datetime.now(timezone.utc).isoformat()

    # Yield initial state with raw data
    yield {
        "type": "init",
        "data": {
            "user_id": user_id,
            "user_name": raw_data.username,
            "raw_system_data": {
                "username": raw_data.username,
                "computer_name": raw_data.computer_name,
                "shell": raw_data.shell,
                "language": raw_data.language,
                "region": raw_data.region,
                "timezone": raw_data.timezone,
                "appearance": raw_data.appearance,
                "installed_apps": raw_data.installed_apps,
                "dev_tools": raw_data.dev_tools,
            },
            "scanned_at": scanned_at,
            "total_topics": len(TOPIC_GENERATION_ORDER),
        }
    }

    # Step 2: Yield basic_info instantly (no LLM)
    basic_info = extract_basic_info_instant(raw_data, user_id)
    basic_info_topic = {
        "topic_id": basic_info.topic_id,
        "topic_name": basic_info.topic_name,
        "icon": basic_info.icon,
        "subtopics": [
            {
                "name": s.name,
                "value": s.value,
                "confidence": s.confidence,
                "evidence": s.evidence,
            }
            for s in basic_info.subtopics
        ]
    }
    generated_topics.append(basic_info_topic)
    yield {"type": "topic", "data": basic_info_topic}

    # Step 3: Generate remaining topics with LLM
    if llm_client is None:
        logger.warning("No LLM client, skipping LLM-based topics")
        # Save partial profile with just basic_info
        _save_profile_data(user_id, raw_data, generated_topics, scanned_at)
        yield {"type": "complete", "data": {"topics_generated": 1}}
        return

    generated_count = 1  # basic_info already done
    for topic_id in TOPIC_GENERATION_ORDER:
        if topic_id == "basic_info":
            continue  # Already done

        try:
            topic_data = await generate_single_topic(topic_id, raw_data, llm_client)

            if topic_data and topic_data.subtopics:
                topic_dict = {
                    "topic_id": topic_data.topic_id,
                    "topic_name": topic_data.topic_name,
                    "icon": topic_data.icon,
                    "subtopics": [
                        {
                            "name": s.name,
                            "value": s.value,
                            "confidence": s.confidence,
                            "evidence": s.evidence,
                        }
                        for s in topic_data.subtopics
                    ]
                }
                generated_topics.append(topic_dict)
                yield {"type": "topic", "data": topic_dict}
                generated_count += 1

        except Exception as e:
            logger.error("Error generating topic %s: %s", topic_id, e)
            yield {
                "type": "error",
                "data": {
                    "topic_id": topic_id,
                    "error": str(e)
                }
            }

    # Save the complete profile to disk
    _save_profile_data(user_id, raw_data, generated_topics, scanned_at)

    yield {
        "type": "complete",
        "data": {
            "topics_generated": generated_count
        }
    }
