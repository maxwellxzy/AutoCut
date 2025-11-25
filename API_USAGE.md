# 错题检测API使用文档

## 概述

本API提供错题检测和题目裁剪功能，返回纯数据和图片URL（不使用base64编码），适合第三方应用集成。

**基础URL**: `http://your-server:5004`

---

## API接口列表

### 1. 错题检测接口

**接口**: `POST /detect`

**功能**: 上传试卷图片，检测错题位置，返回结构化数据和图片URL

#### 请求参数

- **Content-Type**: `multipart/form-data`
- **参数**:
  - `image` (必需): 图片文件（支持 JPG, PNG, JPEG, BMP, TIFF, WEBP）
  - `enable_error_detection` (可选): 是否启用错题检测，默认`true`
    - `true`: 仅裁剪错题
    - `false`: 裁剪所有题目

#### 响应格式

```json
{
  "success": true,
  "data": {
    "session_id": "20250125_143022_a1b2c3d4",
    "error_count": 3,
    "question_count": 10,
    "error_symbol_count": 5,
    "visualization_images": {
      "error_symbols": "/visualizations/20250125_143022_a1b2c3d4/error_symbols.jpg",
      "questions": "/visualizations/20250125_143022_a1b2c3d4/questions.jpg",
      "matched_errors": "/visualizations/20250125_143022_a1b2c3d4/matched_errors.jpg"
    },
    "error_details": [
      {
        "question_box": {
          "bbox": [100, 200, 300, 150]  // [x, y, width, height]
        },
        "error_boxes": [
          {
            "bbox": [150, 250, 30, 30],
            "confidence": 0.95,
            "class_name": "cuo"
          }
        ],
        "match_method": "中心点包含"
      }
    ],
    "crop_results": {
      "mode": "error_only",  // 或 "all_questions"
      "cropped_count": 3,
      "cropped_images": [
        {
          "filename": "error_question_001.jpg",
          "question_id": 1,
          "bbox": [100, 200, 300, 150],
          "error_boxes": [...],
          "match_method": "中心点包含"
        }
      ],
      "zip_url": "/download/20250125_143022_a1b2c3d4"
    }
  }
}
```

#### 调用示例

**Python**:
```python
import requests

url = "http://localhost:5004/detect"

# 上传图片并启用错题检测
with open("test_paper.jpg", "rb") as f:
    files = {"image": f}
    data = {"enable_error_detection": "true"}
    response = requests.post(url, files=files, data=data)

result = response.json()
if result["success"]:
    data = result["data"]
    print(f"检测到 {data['error_count']} 个错题")
    print(f"会话ID: {data['session_id']}")

    # 访问可视化图片
    error_symbols_url = f"http://localhost:5004{data['visualization_images']['error_symbols']}"
    print(f"错误符号检测图: {error_symbols_url}")

    # 下载裁剪结果ZIP
    if data['crop_results']['zip_url']:
        zip_url = f"http://localhost:5004{data['crop_results']['zip_url']}"
        print(f"下载ZIP: {zip_url}")
```

**JavaScript (Fetch)**:
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('enable_error_detection', 'true');

fetch('http://localhost:5004/detect', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(result => {
    if (result.success) {
        const data = result.data;
        console.log(`检测到 ${data.error_count} 个错题`);

        // 显示可视化图片
        document.getElementById('errorImg').src =
            `http://localhost:5004${data.visualization_images.error_symbols}`;
    }
})
.catch(error => console.error('Error:', error));
```

**cURL**:
```bash
curl -X POST http://localhost:5004/detect \
  -F "image=@test_paper.jpg" \
  -F "enable_error_detection=true"
```

---

### 2. 访问可视化图片

**接口**: `GET /visualizations/<session_id>/<filename>`

**功能**: 获取检测过程中生成的可视化图片

#### 参数

- `session_id`: 会话ID（从 `/detect` 接口返回）
- `filename`: 图片文件名，可选值：
  - `error_symbols.jpg` - 错误符号检测图
  - `questions.jpg` - 题目分割图
  - `matched_errors.jpg` - 错题匹配图

#### 示例

```
GET http://localhost:5004/visualizations/20250125_143022_a1b2c3d4/error_symbols.jpg
```

---

### 3. 访问裁剪后的题目图片

**接口**: `GET /files/<session_id>/<filename>`

**功能**: 获取裁剪后的单个题目图片

#### 参数

- `session_id`: 会话ID
- `filename`: 题目图片文件名（从 `crop_results.cropped_images` 中获取）

#### 示例

```
GET http://localhost:5004/files/20250125_143022_a1b2c3d4/error_question_001.jpg
```

---

### 4. 下载所有裁剪结果

**接口**: `GET /download/<session_id>`

**功能**: 下载包含所有裁剪题目的ZIP文件

#### 参数

- `session_id`: 会话ID

#### 示例

**Python**:
```python
import requests

session_id = "20250125_143022_a1b2c3d4"
url = f"http://localhost:5004/download/{session_id}"

response = requests.get(url)
with open("questions.zip", "wb") as f:
    f.write(response.content)
```

**浏览器直接访问**:
```
http://localhost:5004/download/20250125_143022_a1b2c3d4
```

---

## 数据结构说明

### 坐标格式

所有检测框使用 `[x, y, width, height]` 格式：
- `x`: 左上角X坐标
- `y`: 左上角Y坐标
- `width`: 宽度
- `height`: 高度

### 错误符号类型

- `cuo` - 叉号
- `xie` - 斜线
- `bandui` - 半对
- `wenhao` - 问号
- `yuanquan` - 圆圈

### 匹配方法

- `中心点包含` - 错号中心在题目框内
- `重叠面积` - 错号与题目有一定重叠
- `IOU` - 交并比匹配
- `距离最近` - 距离最近的题目
- `兜底匹配` - 强制匹配最近题目

---

## 错误处理

所有接口错误响应格式：

```json
{
  "success": false,
  "message": "错误描述信息"
}
```

### 常见错误码

- `400` - 请求参数错误（如无效的session_id）
- `404` - 资源不存在（如文件不存在）
- `413` - 文件过大（超过16MB）
- `500` - 服务器内部错误

---

## 注意事项

1. **文件大小限制**: 上传图片最大16MB
2. **会话有效期**: 会话文件保留24小时后自动清理
3. **并发处理**: 检测过程较耗时，建议异步处理
4. **URL有效期**: 图片URL在会话有效期内可访问

---

## 完整工作流示例

```python
import requests
import os

# 1. 上传图片进行检测
def detect_errors(image_path):
    url = "http://localhost:5004/detect"

    with open(image_path, "rb") as f:
        files = {"image": f}
        data = {"enable_error_detection": "true"}
        response = requests.post(url, files=files, data=data)

    result = response.json()
    if not result["success"]:
        print(f"检测失败: {result['message']}")
        return None

    return result["data"]

# 2. 处理检测结果
def process_result(data):
    print(f"\n=== 检测结果 ===")
    print(f"会话ID: {data['session_id']}")
    print(f"错题数量: {data['error_count']}")
    print(f"题目总数: {data['question_count']}")
    print(f"错误符号: {data['error_symbol_count']}")

    # 3. 下载可视化图片
    base_url = "http://localhost:5004"
    session_id = data['session_id']

    for name, url in data['visualization_images'].items():
        full_url = f"{base_url}{url}"
        response = requests.get(full_url)

        filename = f"{name}.jpg"
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"已下载: {filename}")

    # 4. 下载所有裁剪结果
    if data['crop_results']['zip_url']:
        zip_url = f"{base_url}{data['crop_results']['zip_url']}"
        response = requests.get(zip_url)

        zip_filename = f"questions_{session_id}.zip"
        with open(zip_filename, "wb") as f:
            f.write(response.content)
        print(f"已下载ZIP: {zip_filename}")

    # 5. 打印错题详情
    print(f"\n=== 错题详情 ===")
    for i, error in enumerate(data['error_details'], 1):
        bbox = error['question_box']['bbox']
        print(f"\n错题 {i}:")
        print(f"  位置: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
        print(f"  匹配方式: {error['match_method']}")
        print(f"  错误符号:")
        for err_box in error['error_boxes']:
            print(f"    - {err_box['class_name']} (置信度: {err_box['confidence']:.2%})")

# 执行
if __name__ == "__main__":
    image_path = "test_paper.jpg"

    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        exit(1)

    data = detect_errors(image_path)
    if data:
        process_result(data)
```

---

## 性能优化建议

1. **批量处理**: 如需处理多张图片，建议使用异步并发调用
2. **图片预处理**: 上传前可适当压缩图片以加快传输
3. **结果缓存**: session_id可用于24小时内重复访问结果
4. **下载优化**: 如只需部分结果，使用 `/files` 接口而非下载完整ZIP

---

## 后续改进计划

- [ ] 添加CORS支持（跨域请求）
- [ ] 添加API Key认证
- [ ] 添加速率限制
- [ ] 提供Webhook回调（异步通知）
- [ ] 支持批量上传
