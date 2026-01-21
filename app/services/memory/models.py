"""
Pydantic models for Memory API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class MemoryTypeEnum(str, Enum):
    """Memory type enumeration for API"""
    EPISODIC = "episodic_memory"
    PROFILE = "profile"
    FORESIGHT = "foresight"
    EVENT_LOG = "event_log"
    GROUP_PROFILE = "group_profile"
    CORE = "core"


class RetrieveMethodEnum(str, Enum):
    """Retrieval method enumeration"""
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"
    RRF = "rrf"
    AGENTIC = "agentic"


# ==================== Request Models ====================

class RawDataItem(BaseModel):
    """Raw data item for memorization"""
    content: Dict[str, Any]
    data_id: str
    data_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemorizeRequest(BaseModel):
    """Request to process and memorize new data"""
    raw_data_list: List[RawDataItem]
    user_id: str
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    enable_foresight: bool = True
    enable_event_log: bool = True


class SearchMemoryRequest(BaseModel):
    """Request to search memories"""
    query: str
    user_id: str
    method: RetrieveMethodEnum = RetrieveMethodEnum.RRF
    memory_types: Optional[List[MemoryTypeEnum]] = None
    limit: int = Field(default=20, ge=1, le=100)


class GetMemoriesRequest(BaseModel):
    """Request to get user memories"""
    user_id: str
    memory_type: Optional[MemoryTypeEnum] = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


# ==================== Response Models ====================

class MemoryRecord(BaseModel):
    """Generic memory record"""
    id: str
    user_id: str
    memory_type: str
    content: str
    summary: Optional[str] = None
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None  # For search results


class EpisodeRecord(BaseModel):
    """Episodic memory record"""
    id: str
    user_id: str
    title: Optional[str] = None
    summary: str
    episode: Optional[str] = None
    timestamp: datetime
    participants: List[str] = []
    subject: Optional[str] = None
    parent_memcell_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemCellRecord(BaseModel):
    """MemCell record - source data for memory extraction"""
    id: str
    user_id: str
    original_data: str  # JSON string
    summary: Optional[str] = None
    subject: Optional[str] = None
    file_id: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_ordinal: Optional[int] = None
    type: Optional[str] = None  # 'Document' or 'Conversation'
    keywords: Optional[List[str]] = None
    timestamp: datetime
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class MemCellDetail(BaseModel):
    """MemCell with linked episodes and event logs"""
    memcell: MemCellRecord
    episodes: List[EpisodeRecord] = []
    event_logs: List['EventLogRecord'] = []


class ProfileRecord(BaseModel):
    """User profile record"""
    user_id: str
    user_name: Optional[str] = None
    personality: Optional[List[str]] = None
    hard_skills: Optional[List[Dict[str, str]]] = None
    soft_skills: Optional[List[Dict[str, str]]] = None
    interests: Optional[List[str]] = None
    motivation_system: Optional[List[Dict[str, Any]]] = None
    value_system: Optional[List[Dict[str, Any]]] = None
    projects_participated: Optional[List[Dict[str, str]]] = None
    updated_at: Optional[datetime] = None


class ForesightRecord(BaseModel):
    """Foresight/prospective memory record"""
    id: str
    user_id: str
    content: str
    evidence: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_days: Optional[int] = None
    parent_episode_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EventLogRecord(BaseModel):
    """Event log (atomic fact) record"""
    id: str
    user_id: str
    atomic_fact: str
    timestamp: datetime
    parent_episode_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemorizeResult(BaseModel):
    """Result of memorization process"""
    success: bool
    message: str
    episodes_created: int = 0
    event_logs_created: int = 0
    foresights_created: int = 0
    profile_updated: bool = False


class SearchMemoryResult(BaseModel):
    """Result of memory search"""
    memories: List[MemoryRecord]
    total_count: int
    query: str
    method: str


class UserMemorySummary(BaseModel):
    """Summary of user's memories"""
    user_id: str
    profile: Optional[ProfileRecord] = None
    memcells_count: int = 0
    episodes_count: int = 0
    event_logs_count: int = 0
    foresights_count: int = 0
    recent_episodes: List[EpisodeRecord] = []
    recent_foresights: List[ForesightRecord] = []


# ==================== Basic Profile Models (LLM-Inferred from System) ====================

class SkillRecord(BaseModel):
    """Skill with level"""
    name: str
    level: Optional[str] = None  # beginner/intermediate/advanced


class RawSystemDataRecord(BaseModel):
    """Raw system data collected from Mac"""
    username: str = ""
    computer_name: str = ""
    shell: str = ""
    language: str = ""
    region: str = ""
    timezone: str = ""
    appearance: str = ""
    installed_apps: List[str] = []
    dev_tools: List[Dict[str, str]] = []


class ProfileSubtopic(BaseModel):
    """
    A single subtopic field within a profile topic.

    Example:
        name: "language_spoken"
        value: ["English", "Chinese"]
        confidence: "high"
        evidence: "System language is en-US, region is CN"
    """
    name: str
    description: Optional[str] = None
    value: Optional[Any] = None  # Can be str, List[str], int, etc.
    confidence: Optional[str] = None  # "high", "medium", "low", "inferred"
    evidence: Optional[str] = None  # What data was used to infer this


class ProfileTopic(BaseModel):
    """
    A topic category containing multiple subtopics.

    Example:
        topic_id: "basic_info"
        topic_name: "Basic Information"
        icon: "user"
        subtopics: [ProfileSubtopic(...), ...]
    """
    topic_id: str
    topic_name: str
    icon: Optional[str] = None  # Icon name for UI
    subtopics: List[ProfileSubtopic] = []


# Default profile topics schema (user can extend this)
DEFAULT_PROFILE_TOPICS = [
    {
        "topic_id": "basic_info",
        "topic_name": "Basic Information",
        "icon": "user",
        "subtopics": ["name", "age", "gender", "birth_date", "nationality", "ethnicity", "language_spoken"]
    },
    {
        "topic_id": "contact_info",
        "topic_name": "Contact Information",
        "icon": "mail",
        "subtopics": ["email", "phone", "city", "country"]
    },
    {
        "topic_id": "education",
        "topic_name": "Education",
        "icon": "graduation-cap",
        "subtopics": ["school", "degree", "major"]
    },
    {
        "topic_id": "work",
        "topic_name": "Work & Career",
        "icon": "briefcase",
        "subtopics": ["company", "title", "working_industry", "work_skills", "work_responsibility"]
    },
    {
        "topic_id": "interest",
        "topic_name": "Interests & Hobbies",
        "icon": "heart",
        "subtopics": ["books", "movies", "music", "foods", "sports", "hobbies"]
    },
    {
        "topic_id": "psychological",
        "topic_name": "Psychological Profile",
        "icon": "brain",
        "subtopics": ["personality", "values", "beliefs", "motivations", "goals", "fears"]
    },
    {
        "topic_id": "behavioral",
        "topic_name": "Behavioral Patterns",
        "icon": "activity",
        "subtopics": ["working_habit", "decision_making_style", "communication_style", "humor_style"]
    },
    {
        "topic_id": "technical",
        "topic_name": "Technical Profile",
        "icon": "code",
        "subtopics": ["hard_skills", "soft_skills", "tools_used", "preferred_technologies"]
    },
]


class BasicProfileResponse(BaseModel):
    """
    Hierarchical LLM-inferred semantic profile from Mac system data.

    Structure:
    - topics: List of ProfileTopic, each containing subtopics
    - raw_system_data: The raw Mac system data used for inference

    This profile is inferred by LLM from:
    - Installed applications
    - Development tools
    - System preferences (language, region, appearance)
    - User information (username, computer name)
    """
    user_id: str
    user_name: Optional[str] = None

    # Hierarchical profile structure
    topics: List[ProfileTopic] = []

    # Legacy flat fields (for backward compatibility)
    personality: List[str] = []
    interests: List[str] = []
    hard_skills: List[SkillRecord] = []
    soft_skills: List[SkillRecord] = []
    working_habit_preference: List[str] = []
    user_goal: List[str] = []
    motivation_system: List[str] = []
    value_system: List[str] = []
    inferred_roles: List[str] = []

    # Raw data for reference
    raw_system_data: Optional[RawSystemDataRecord] = None
    scanned_at: str  # ISO timestamp
