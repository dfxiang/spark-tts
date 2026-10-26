# 声音管理

## 名称索引

服务启动时扫描 `SPARK_TTS_VOICE_DIR`，根据每个声音目录中的
`metadata.json` 建立：

```text
voice ID -> VoiceRecord
normalized display name -> [voice ID]
```

生成接口的 `voice` 参数可以先按 ID 查找，再按显示名称查找。名称比较会忽略首尾
空白和 ASCII 大小写。

- 名称唯一时，可以直接传名称。
- 名称重复时返回 `409 voice_name_ambiguous`，要求改用 voice ID。
- 新建声音时不允许使用已有名称。
- 新建声音未填写名称时，默认使用上传文件名，支持中文和原扩展名；
  文件名重复时会自动追加 ` (2)`、` (3)`。
- 修改历史重名声音的其它 metadata 时，可以保留当前名称。

示例：

```json
{
  "input": "需要合成的文本",
  "voice": "客服女声",
  "response_format": "wav"
}
```

## API

| 方法 | 路径 | 功能 |
|---|---|---|
| `GET` | `/api/voices` | 列出声音 |
| `POST` | `/api/voices` | 上传参考音频并创建声音 |
| `GET` | `/api/voices/{voice}` | 获取声音 metadata |
| `PATCH` | `/api/voices/{voice}` | 修改 `name` 或 `prompt_text` |
| `GET` | `/api/voices/{voice}/audio` | 获取参考音频 WAV |
| `PUT` | `/api/voices/{voice}/audio` | 替换参考音频 |
| `DELETE` | `/api/voices/{voice}` | 删除声音及 metadata |

`PATCH` 示例：

```json
{
  "name": "新的显示名称",
  "prompt_text": "参考音频对应的准确文字"
}
```

管理路径中的 `{voice}` 使用 voice ID，避免修改名称后路径失效。

## 管理页面

访问 `/voices` 可以使用独立的声音管理页面：

- 上传新声音。
- 查看名称、ID、参考音频时长和创建时间。
- 修改显示名称和参考文本。
- 替换参考音频。
- 加载试听参考音频。
- 删除声音。

页面主入口位于语音合成页右上角的“声音管理”。

语音合成页本身不再直接上传参考音频，而是通过下拉框选择声音；选择后会展示
显示名称、voice ID、参考音频时长、创建时间和参考文本，并自动加载参考音频试听。
