from __future__ import annotations

import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

from .service import EmailAccountNotFound, EmailAuthError, EmailService, EmailServiceError, EmailSyncError, get_email_service
from .models import (
    EmailAccountCreate,
    EmailAccountSummary,
    EmailMessageContent,
    EmailMessageSummary,
    EmailSyncRequest,
    EmailSyncResult,
)

router = APIRouter(tags=["plugin-mail"])


# ==================== Account-Level Memory Integration Models (v2.5) ====================

class BuildAccountMemoryRequest(BaseModel):
    """Request to build memory for an email account"""
    user_id: str = "default_user"
    force: bool = False  # If True, rebuild all emails even if already processed


class BuildAccountMemoryResponse(BaseModel):
    """Response from account memory building"""
    success: bool
    message: str
    account_id: str
    total_messages: int = 0
    memcells_created: int = 0
    episodes_created: int = 0
    event_logs_created: int = 0


class AccountQARequest(BaseModel):
    """Request to ask a question about account emails"""
    question: str
    user_id: str = "default_user"


class AccountQAResponse(BaseModel):
    """Response from account QA"""
    answer: str
    sources: List[dict] = []  # Related memories/context used


class MemCellItem(BaseModel):
    """A single MemCell from email memory"""
    id: str
    email_subject: str
    email_sender: Optional[str] = None
    preview: Optional[str] = None
    timestamp: Optional[str] = None


class EpisodeItem(BaseModel):
    """A single episode from email memory"""
    id: str
    memcell_id: Optional[str] = None
    email_subject: Optional[str] = None
    summary: str
    episode: Optional[str] = None
    timestamp: Optional[str] = None


class FactItem(BaseModel):
    """A single atomic fact from email memory"""
    id: str
    episode_id: Optional[str] = None
    email_subject: Optional[str] = None
    fact: str
    timestamp: Optional[str] = None


class AccountMemoryDetailsResponse(BaseModel):
    """Detailed memory items for an account"""
    account_id: str
    memcells: List[MemCellItem] = []
    episodes: List[EpisodeItem] = []
    facts: List[FactItem] = []
    total_memcells: int = 0
    total_episodes: int = 0
    total_facts: int = 0


class AccountMemoryStatusResponse(BaseModel):
    """Memory status for an email account"""
    account_id: str
    is_built: bool
    memcell_count: int = 0
    episode_count: int = 0
    event_log_count: int = 0
    last_built_at: Optional[str] = None


class OutlookAuthRequest(BaseModel):
    client_id: str
    tenant_id: str


class OutlookCompleteRequest(BaseModel):
    flow_id: str
    label: str


@router.post("/outlook/auth", response_model=dict)
async def start_outlook_auth(
    payload: OutlookAuthRequest,
    service: EmailService = Depends(get_email_service)
) -> dict:
    try:
        flow_id = await service.start_outlook_auth(payload.client_id, payload.tenant_id)
        return {"flow_id": flow_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/outlook/auth/{flow_id}", response_model=dict)
async def get_outlook_auth_status(
    flow_id: str,
    service: EmailService = Depends(get_email_service)
) -> dict:
    try:
        return await service.get_outlook_auth_status(flow_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/outlook/complete", response_model=EmailAccountSummary)
async def complete_outlook_setup(
    payload: OutlookCompleteRequest,
    service: EmailService = Depends(get_email_service)
) -> EmailAccountSummary:
    try:
        return await service.complete_outlook_setup(payload.flow_id, payload.label)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/accounts", response_model=list[EmailAccountSummary])
async def list_accounts(service: EmailService = Depends(get_email_service)) -> list[EmailAccountSummary]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, service.list_accounts)


@router.post("/accounts", response_model=EmailAccountSummary, status_code=status.HTTP_201_CREATED)
async def add_account(
    payload: EmailAccountCreate,
    service: EmailService = Depends(get_email_service),
) -> EmailAccountSummary:
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: service.add_account(payload))
    except EmailServiceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.delete("/accounts/{account_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def remove_account(account_id: str, service: EmailService = Depends(get_email_service)) -> Response:
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: service.remove_account(account_id))
    except EmailAccountNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/accounts/{account_id}/sync", response_model=EmailSyncResult)
async def sync_account(
    account_id: str,
    request: EmailSyncRequest,
    service: EmailService = Depends(get_email_service),
) -> EmailSyncResult:
    try:
        return await service.sync_account(account_id, request)
    except EmailAccountNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EmailAuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except EmailSyncError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except EmailServiceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/accounts/{account_id}/messages", response_model=list[EmailMessageSummary])
def list_messages(
    account_id: str,
    limit: int = Query(default=50, ge=1, le=EmailService.MAX_SYNC_BATCH),
    service: EmailService = Depends(get_email_service),
) -> list[EmailMessageSummary]:
    try:
        return service.list_messages(account_id, limit)
    except EmailAccountNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/messages/{message_id}", response_model=EmailMessageContent)
def get_message(message_id: str, service: EmailService = Depends(get_email_service)) -> EmailMessageContent:
    try:
        return service.get_message(message_id)
    except EmailServiceError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


# ==================== Account-Level Memory Integration Endpoints (v2.5) ====================

@router.post("/accounts/{account_id}/build-memory", response_model=BuildAccountMemoryResponse)
async def build_account_memory(
    account_id: str,
    request: BuildAccountMemoryRequest,
    service: EmailService = Depends(get_email_service),
) -> BuildAccountMemoryResponse:
    """
    一键构建：为整个邮箱账户批量构建 Memory
    
    遍历该账户下的所有邮件，为每封邮件创建 MemCell、Episode 和 EventLog。
    """
    try:
        result = await service.build_account_memory(account_id, request.user_id)
        return BuildAccountMemoryResponse(**result)
    except EmailAccountNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/accounts/{account_id}/build-memory/stream")
async def build_account_memory_stream(
    account_id: str,
    request: BuildAccountMemoryRequest,
    service: EmailService = Depends(get_email_service),
):
    """
    流式构建 Memory：实时报告进度和每封邮件的提取结果
    
    返回 SSE (Server-Sent Events) 格式的进度更新。
    
    Args:
        force: 如果为 True，强制重建所有邮件的 Memory（忽略已打标的邮件）
    """
    import json
    
    async def generate_progress():
        try:
            async for progress in service.build_account_memory_stream(account_id, request.user_id, request.force):
                yield f"data: {json.dumps(progress, ensure_ascii=False)}\n\n"
        except EmailAccountNotFound as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/accounts/{account_id}/memory-status", response_model=AccountMemoryStatusResponse)
async def get_account_memory_status(
    account_id: str,
    user_id: str = Query(default="default_user"),
    service: EmailService = Depends(get_email_service),
) -> AccountMemoryStatusResponse:
    """
    获取邮箱账户的 Memory 状态
    """
    try:
        result = await service.get_account_memory_status(account_id, user_id)
        return AccountMemoryStatusResponse(**result)
    except EmailAccountNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/accounts/{account_id}/qa", response_model=AccountQAResponse)
async def account_qa(
    account_id: str,
    request: AccountQARequest,
    service: EmailService = Depends(get_email_service),
) -> AccountQAResponse:
    """
    邮箱账户问答：基于该账户的所有邮件记忆进行问答
    """
    try:
        result = await service.account_qa(account_id, request.question, request.user_id)
        return AccountQAResponse(**result)
    except EmailAccountNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EmailServiceError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get("/accounts/{account_id}/memory-details", response_model=AccountMemoryDetailsResponse)
async def get_account_memory_details(
    account_id: str,
    user_id: str = Query(default="default_user"),
    limit: int = Query(default=50, ge=1, le=200),
    service: EmailService = Depends(get_email_service),
) -> AccountMemoryDetailsResponse:
    """
    获取邮箱账户的 Memory 详情：Episodes 和 Facts 列表
    
    用于可视化展示已提取的情节记忆和原子事实
    """
    try:
        result = await service.get_account_memory_details(account_id, user_id, limit)
        return AccountMemoryDetailsResponse(**result)
    except EmailAccountNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


class FailedEmailItem(BaseModel):
    """Failed email item"""
    id: str
    subject: Optional[str] = None
    sender: Optional[str] = None
    error: Optional[str] = None
    failed_at: Optional[str] = None


class FailedEmailsResponse(BaseModel):
    """Response with failed emails list"""
    account_id: str
    failed_emails: List[FailedEmailItem] = []
    total_failed: int = 0


class RetryEmailRequest(BaseModel):
    """Request to retry a specific email"""
    user_id: str = "default_user"


@router.get("/accounts/{account_id}/failed-emails")
async def get_failed_emails(
    account_id: str,
    service: EmailService = Depends(get_email_service),
) -> FailedEmailsResponse:
    """
    获取打标失败的邮件列表
    """
    try:
        failed = service.storage.list_failed_email_messages(account_id)
        return FailedEmailsResponse(
            account_id=account_id,
            failed_emails=[
                FailedEmailItem(
                    id=msg.id,
                    subject=msg.subject,
                    sender=msg.sender,
                    error=msg.memory_error,
                    failed_at=msg.memory_built_at.isoformat() if msg.memory_built_at else None,
                )
                for msg in failed
            ],
            total_failed=len(failed),
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/accounts/{account_id}/retry-email/{message_id}")
async def retry_failed_email(
    account_id: str,
    message_id: str,
    request: RetryEmailRequest,
    service: EmailService = Depends(get_email_service),
):
    """
    重试单个失败的邮件打标
    
    返回 SSE 格式的进度更新
    """
    import json
    
    async def generate_progress():
        try:
            async for progress in service.retry_single_email(account_id, message_id, request.user_id):
                yield f"data: {json.dumps(progress, ensure_ascii=False)}\n\n"
        except EmailAccountNotFound as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
