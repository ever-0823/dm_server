import os
import io
import csv
from ..core.responses import success_response
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, UploadFile, File, Query
from fastapi.responses import FileResponse
from ..core.config import Settings
from ..core.exceptions import AppException
from ..dependencies.auth import current_user
from ..model.device import Device
from ..schems.device import DeviceBatchDeleteRequest, DeviceCreateRequest, DeviceUpdateRequest
from ..model.device_log import DeviceLog
from typing import Literal

router = APIRouter()


def get_existing_device(device_id: str):
    existing = Device.find_by_device_id(device_id)
    if not existing:
        raise AppException(404, "设备不存在")
    else:
        return existing


"""
Depends(current_user)：进入接口前，先校验登录态
"""

"""分页查询"""


@router.get("/devices")
def get_devices(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100), search: str = Query(""),
                status: Literal["", "active", "inactive", "maintenance", "retired"] = "", user=Depends(current_user)):
    # devices = Device.get_all()
    print(f"获取到的关键词:{search}")
    print(f"获取到的设备状态:{status}")
    # result = Device.get_list(page=page, page_size=page_size, search=search)
    result = Device.get_list(page=page, page_size=page_size, search=search, status=status)
    return success_response(
        data=result,
        current_user=user["username"],
    )


"""新增设备"""

@router.post("/devices")
def create_device(payload: DeviceCreateRequest, user=Depends(current_user)):
    # 先检查device_id是否重复
    existing = Device.find_by_device_id(payload.device_id)
    if existing:
        raise AppException(400, "设备编号已存在")
    # model_dump() 转成字典
    device_id = Device.create(payload.model_dump())
    print(f"device_id:{device_id}")
    print(f"operation:{user['username']}")
    DeviceLog.create(
        payload.device_id, "create", operation=user["username"], details=f"创建设备:{payload.device_name}"
    )

    return success_response(
        message="设备创建成功",
        id=device_id,
        operator=user["username"],
    )


"""设备状态统计"""


@router.get("/devices/statistics")
def get_device_statistics(user=Depends(current_user)):
    # print(f"设备状态统计")
    result = Device.get_statistics()
    return success_response(
        data=result,
        current_user=user["username"],
    )


"""CSV 导出接口"""


@router.get("/devices/export")
def export_devices(user=Depends(current_user)):
    devices = Device.get_all_for_export()
    # io.StringIO()：在内存里创建一个“文本文件”。
    # csv.writer(output)：把 CSV 内容写到这个内存文件里。
    output = io.StringIO()
    write = csv.writer(output)

    # 导出表头和设备列表页保持一致，避免界面列名和 CSV 列名不统一。
    write.writerow(["设备编号", "设备名称", "型号", "厂商", "位置", "状态"])
    for device in devices:
        write.writerow([
            device["device_id"],
            device["device_name"],
            device["model"],
            device["manufacturer"],
            device["location"],
            device["status"]
        ])
    output.seek(0)
    DeviceLog.create(
        "All",
        "export",
        user["username"],
        "导出设备列表 CSV",
    )
    # Excel 直接打开 CSV 时通常会依赖 BOM 识别 UTF-8，这里补上以避免中文乱码。
    csv_bytes = output.getvalue().encode("utf-8-sig")
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=devices.csv"},
    )


"""CSV导入接口"""


@router.post("/devices/import")
def import_devices(file: UploadFile = File(...), user=Depends(current_user)):
    # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 UTF-8，便于导出文件再直接导回。
    content = file.file.read().decode("utf-8-sig")
    # csv.DictReader(...)：按“表头 -> 值”的方式逐行读取。
    reader = csv.DictReader(io.StringIO(content))

    success_count = 0
    skipped_count = 0

    # 导入同时兼容旧英文表头和新中文表头，避免导出格式调整后影响回传导入。
    column_aliases = {
        "device_id": ["device_id", "设备编号"],
        "device_name": ["device_name", "设备名称"],
        "model": ["model", "型号"],
        "manufacturer": ["manufacturer", "厂商"],
        "location": ["location", "位置"],
        "status": ["status", "状态"],
    }

    # .strip() 是 Python 字符串的内置方法，用于移除字符串首尾的空白字符。
    for row in reader:
        # 按别名顺序取值，先命中的列名直接使用。
        normalized_row = {
            field: next((str(row.get(alias, "")).strip() for alias in aliases if str(row.get(alias, "")).strip()), "")
            for field, aliases in column_aliases.items()
        }
        device_id = normalized_row["device_id"]
        device_name = normalized_row["device_name"]

        if not device_id or not device_name:
            skipped_count += 1
            continue

        existing = Device.find_by_device_id(device_id)
        if existing:
            skipped_count += 1
            continue

        Device.create(
            {
                "device_id": device_id,
                "device_name": device_name,
                "model": normalized_row["model"],
                "manufacturer": normalized_row["manufacturer"],
                "location": normalized_row["location"],
                "status": normalized_row["status"] or "active",
            }
        )
        success_count += 1

    # 整个 CSV 处理完成后再写一条汇总日志，避免循环内提前返回。
    DeviceLog.create(
        "ALL",
        "import",
        user["username"],
        f"导入设备 CSV，成功 {success_count} 条，跳过 {skipped_count} 条",
    )

    return success_response(
        message="导入完成",
        data={
            "success_count": success_count,
            "skipped_count": skipped_count,
        },
        operator=user["username"],
    )


"""
查询单个设备详情

"""


@router.get("/devices/{device_id}")
def get_device_detail(device_id: str, user=Depends(current_user)):
    device = get_existing_device(device_id)
    # print(f"查询单个设备详情:{device.get('file_path')}")
    logs = DeviceLog.get_device_id(device_id)

    return success_response(
        data={
            "device": device,
            "logs": logs,
        },
        current_user=user["username"],
    )


"""删除设备"""


@router.delete("/devices/{device_id}")
def delete_device(device_id: str, user=Depends(current_user)):
    existing = get_existing_device(device_id)
    Device.delete_by_device_id(device_id)
    DeviceLog.create(
        device_id, "delete", operation=user["username"], details=f"删除设备:{existing['device_name']}"
    )
    # return {
    #     "success": True,
    #     "message": "设备删除成功",
    #     "operator": user["username"],
    # }
    return success_response(
        message="设备删除成功",
        operator=user["username"],
    )


@router.post("/devices/batch-delete")
def batch_delete_devices(payload: DeviceBatchDeleteRequest, user=Depends(current_user)):
    existing_devices = []
    missing_ids = []

    for device_id in payload.device_ids:
        existing = Device.find_by_device_id(device_id)
        if existing:
            existing_devices.append(existing)
        else:
            missing_ids.append(device_id)

    if not existing_devices:
        raise AppException(404, "未找到可删除的设备")

    deleted_ids = [device["device_id"] for device in existing_devices]
    deleted_count = Device.delete_by_device_ids(deleted_ids)

    for device in existing_devices:
        DeviceLog.create(
            device["device_id"],
            "batch_delete",
            operation=user["username"],
            details=f"批量删除设备:{device['device_name']}",
        )

    return success_response(
        message="批量删除完成",
        data={
            "deleted_count": deleted_count,
            "deleted_ids": deleted_ids,
            "missing_ids": missing_ids,
        },
        operator=user["username"],
    )


"""根据device_id修改设备信息"""


@router.put("/devices/{device_id}")
def update_device(device_id: str, payload: DeviceUpdateRequest, user=Depends(current_user)):
    get_existing_device(device_id)
    Device.update_by_device_id(device_id, payload.model_dump())
    DeviceLog.create(
        device_id, "update", operation=user["username"], details=f"更新设备:{payload.device_name}"
    )
    return success_response(
        message="设备更新成功",
        operator=user["username"],
    )


@router.get("/devices/{device_id}/logs")
def get_device_logs(device_id: str, user=Depends(current_user)):
    get_existing_device(device_id)
    logs = DeviceLog.get_device_id(device_id)
    return success_response(
        message="设备更新成功",
        data=logs,
        operator=user["username"],
    )


"""设备附件上传"""


@router.post("/devices/{device_id}/upload")
def upload_device_file(device_id: str, file: UploadFile = File(...), user=Depends(current_user)):
    get_existing_device(device_id)

    os.makedirs(Settings.UPLOAD_FOLDER, exist_ok=True)
    filename = f"{device_id}_{file.filename}"
    file_path = os.path.join(Settings.UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    Device.update_file_path(device_id, file_path)
    DeviceLog.create(
        device_id, "upload", user["username"], f"上传附件:{file.filename}"
    )
    # return {
    #     "success": True,
    #     "message": "文件上传成功",
    #     "filename": file.filename,
    #     "file_path": file_path,
    # }
    return success_response(
        message="文件上传成功",
        data={
            "filename": file.filename
        },
        operator=user["username"],
    )


"""附件下载"""


@router.get("/devices/{device_id}/download")
def download_device_file(device_id: str, user=Depends(current_user)):
    device = get_existing_device(device_id)
    file_path = str(device.get("file_path"))
    if not file_path:
        raise AppException(404, "该设备没有附件")
    if not os.path.exists(file_path):
        raise AppException(404, "附件文件不存在")
    DeviceLog.create(
        device_id, "download", user["username"], f"下载附件:{os.path.basename(file_path)}"
    )
    return FileResponse(file_path, filename=os.path.basename(file_path))


"""删除从设备"""


@router.post("/devices/{device_id}/delete")
def delete_device(device_id: str, user=Depends(current_user)):
    device = get_existing_device(device_id)
    file_path = str(device.get("file_path"))
    if not file_path:
        return {
            "success": False,
            "message": "该设备没有附件"
        }
    # 查询文件目录是否存在
    if os.path.exists(file_path):
        os.remove(file_path)
    Device.update_file_path(device_id, "")
    DeviceLog.create(
        device_id,
        "delete_file",
        user["username"],
        "删除附件"
    )
    # return {
    #     "success": True,
    #     "message": "附件删除成功",
    #     "operator": user["username"],
    # }
    return success_response(
        message="附件删除成功",
        operator=user["username"],
    )


