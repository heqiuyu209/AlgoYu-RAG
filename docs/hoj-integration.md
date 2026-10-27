# HOJ 内部索引适配器

此适配器仅供 HOJ 服务端调用。HOJ 负责登录、管理员维护权限、资料发布 / 可见性、持久入库任务及正式 / 虚拟参赛限制；本服务不提供用户登录或聊天模型接口。2026-10-19 已接通 HOJ 资料业务、Vue 页面与常驻 CPU 服务，并验证真实 Markdown / TXT 上传、显式发布、检索 / 原文及重启保留；用户模型生成与固定问答集质量仍待验收。

代码位于 `src/learning_os/hoj`，与个人 CLI / Streamlit 的 collection 和 `.env` 分开。复用现有 Markdown 解析、递归切片和本地 BGE-M3；文本格式直接切片。PDF 支持本地原有文字提取 / OCR 与物理页引用，视频支持本地 ASR / 字幕校订与时间引用；HOJ 保留原始资料并校验下载 / 播放权限。当前质量、部署和未完成条件以 [质量检查点](hoj-evaluation.md) 为准。

发布代码只包含业务模块与对应测试；`experiments/`、本地 `data/` 和模型不随发布提交收录。质量文档中保留的历史实验路径用于本机接续，不表示这些未合格候选已进入生产。九组冻结评测资料保留原始字节；Git属性禁止自动转换其换行，不能以重算freeze掩盖检出变化。

## 运行

HOJ 服务使用 Python >= 3.12，安装项目的 `hoj` 可选依赖；固定视频依赖 av19要求3.12，原个人CLI仍保留3.11支持。需先由运维安装完整 BGE-M3 模型；服务仅接受本地目录，并强制离线模式，启动时不下载模型或加载个人 `.env`。首次加载较慢；工厂完成本地预热后才接受业务请求。

```powershell
python -m pip install -e '.[hoj]'
$env:HOJ_RAG_MODEL_PATH='D:\已安装的模型目录'
$env:HOJ_RAG_DATA='D:\专用索引目录'
$env:HOJ_RAG_DEVICE='cpu'
# 优先使用只读服务凭据文件，内容为随机 32 字节的小写 hex；不要提交或打印。
$env:HOJ_RAG_SHARED_KEY_FILE='D:\专用凭据目录\service.key'
python -m uvicorn learning_os.hoj.service:from_env --factory --host 127.0.0.1 --port 8097 --workers 1
```

每个数据目录仅由一个 worker 持有。索引使用独立 `hoj_knowledge_v1` collection；SQLite manifest 保存 WRITING / READY / DELETED 及短期防重放记录。这里只记录索引状态，不记录用户密钥或替代 HOJ ACL。向量入库相对浏览器必须异步：HOJ 持久任务调用本服务同步入库，收到完整回执后再验证并发布版本；PDF 的限时文字提取在上传请求内完成，成功后原件 / 正文 / 持久任务一同保存。Embedding 进程受限于本地设备，最多两个请求进入执行 / 等待，超时结果不能发布。

不要将此端口映射给浏览器或公网，也不要把用户模型 Key 发给此服务。HOJ Java 使用固定 `hoj.rag.url` origin 和 `hoj.rag.shared-key-file`，共享凭据由运维分别注入两端；为空时保持禁用。Python 兼容 HOJ_RAG_SHARED_KEY 环境注入，但不能同时设置该值与 HOJ_RAG_SHARED_KEY_FILE；推荐文件挂载，读取错误会阻止启动，不生成替代密钥。

### HOJ 本地 Docker 入口

`Dockerfile.hoj` 使用 Python 3.12、测试过的 requirements-hoj.txt 和 CPU PyTorch，运行用户 10001。构建仅包含 src / 运行脚本，不包含原型 `.env`、个人数据、模型或 pyc；模型由外部只读挂载，启动离线预热。`scripts/init_hoj_identity.py` 仅用于一次性初始化卷内独立服务凭据（保留已有有效文件），`scripts/hoj_healthcheck.py` 使用签名空范围检索检查就绪，不占用向量计算容量。

HOJ 仓库 `D:\IdeaProjects\HOJ-master\HOJ-master\compose.rag.yaml` 作为第三份 overlay 提供初始化容器、内部服务、索引卷 / 凭据卷及后端只读凭据挂载。Git 根目录的本地环境文件 HOJ-master/.env.rag.local 只保存 HOJ_RAG_SOURCE_DIR 与 HOJ_RAG_MODEL_DIR 路径，实际本地文件已忽略；不把服务密钥或用户 Key 放在路径文件中。

```powershell
# 在 HOJ Git 根目录；先按 HOJ 的 sqlAndsetting/README.md 完成已有库增量迁移。
docker compose --env-file HOJ-master/.env.rag.local -p hoj-master `
  -f HOJ-master/compose.yaml -f HOJ-master/compose.dev.yaml -f HOJ-master/compose.rag.yaml `
  build hoj-rag-identity hoj-backend
docker compose --env-file HOJ-master/.env.rag.local -p hoj-master `
  -f HOJ-master/compose.yaml -f HOJ-master/compose.dev.yaml -f HOJ-master/compose.rag.yaml `
  up -d
```

服务只在 Docker 内监听 8097，不映射主机端口；只读挂载模型和凭据，/data 使用独立持久卷且每目录一个 worker。后续 `up` 继续带齐三份 Compose，防止撤掉后端配置；重建镜像保留两个命名卷。实际镜像更新确认凭据未变化，实际有测试向量时重启确认点位和发布版本可用且不重新入库。本地 CPU 结果不代表生产吞吐或多副本容量。

### 可选同资料片段选择

安装固定cross-encoder/mmarco-mMiniLMv2-L12-H384-v1@1427fd652930e4ba29e8149678df786c240d8825完整七文件，文件清单/尺寸/SHA256以src/learning_os/hoj/reranker.py为信任根；服务不会自动下载或信任安装manifest。额外配置/代码、缺文件/摘要错误或不兼容架构使显式启用启动失败。仅支持dense组合；未设置HOJ_RAG_RERANK_MODEL_PATH时保留原检索，启用后先预热再监听，CPU执行。模型是本地分类器，不接收用户Key，也不调用用户聊天模型。

在本地忽略文件.env.rag.local加入HOJ_RAG_RERANK_MODEL_DIR（已安装模型目录），后续Compose保持base/dev/rag/video/rag-rerank五份overlay，防止撤下已启用的视频/模型挂载。本地更新仅需已有合格镜像的hoj-rag，保留数据/身份/媒体卷；本轮无索引代次或SQL变化，不必重编码。

```powershell
docker compose --env-file HOJ-master/.env.rag.local -p hoj-master `
  -f HOJ-master/compose.yaml -f HOJ-master/compose.dev.yaml -f HOJ-master/compose.rag.yaml `
  -f HOJ-master/compose.video.yaml -f HOJ-master/compose.rag-rerank.yaml up -d --no-deps hoj-rag
```

该选择器保留首两条、主资料全部槽位和完整文档/版本顺序，只在授权语义候选内选择同一次要资料的原片段。同一次要来源有m个可替换槽位时，保留已有候选，再按原cosine补至min(n,m+1)；topK=5最多forward六对，只有实际预算内分数参与排名，未forward的零占位不能压过真实负分。唯一候选正是原片段时不forward；完整原pool（包括排除/固定项）仍tokenize以保留512整次回退。公开分数仍原cosine（不保证数值递减），引用由HOJ核对；CE logit不公开，也不作答案正确率或自动拒答阈值。模型执行不持存储锁，返回前新读READY/哈希，删除期间迟到结果409。

本机现已启用以下上下文投影，实际更新使用下节的六份 Compose 命令；上面的五份命令只适用于单索引配置。

### 可选上下文投影与 ONNX 执行器

2026-10-26 已完成业务工厂接入、349 项主机检查、真实工厂/TCP及完整300问独立审计，并部署本地c1517e20…；完整集成质量/耗时和剩余问题以[质量检查点](hoj-evaluation.md)为准。默认配置仍使用原路径。显式启用 `HOJ_RAG_RERANK_ENGINE=onnx` 时还须设置本地 `HOJ_RAG_RERANK_GRAPH_PATH`；图的固定尺寸/SHA256 由 `onnx_reranker.py` 校验，原七文件及分词器仍须通过身份检查。仅 CPU、FP32、物理 batch2，保留完整候选池 512 token 回退、共享 inference gate 和绝对截止时间。配置错误或加载失败直接阻止启动。

`HOJ_RAG_CONTEXT_PROJECTION=1` 严格要求 dense＋ONNX。raw collection/manifest 继续作为可见性权威，辅助表示使用独立 `hoj_knowledge_context_r1` / `context-r1.sqlite3`。组合导入在两份 READY 后才返回 INDEXED；canonical 切片和完整 embedding 输入一致时复用已校验的 raw 向量，只编码发生变化的输入。删除先提交两份 tombstone 再物理清理。已有 raw READY 不会自动拥有辅助表示，可对同版本再次导入补建；缺少辅助表示时仍可按 raw 检索。

本地已保留原五份 Compose并附加 `compose.rag-context.yaml`，后续更新继续带齐六份。额外路径变量 `HOJ_RAG_RERANK_ONNX_DIR` 指向包含固定 `model_O2.onnx` 的目录，已保存在忽略的 `.env.rag.local`；该目录只读挂载到 `/models/mmarco-onnx`，不放入镜像或改写原模型。实际七挂载/三命名卷和身份核对保留，后端/判题镜像未变。真实工厂/TCP探针已验证签名、重放、截止、空权限、相同输入复用、真实分词512/513，以及评分 gate 占用时删除并拒绝迟到结果；不将人工阻塞调度描述成原生 ORT kernel 计算中断。

### 可选 BGE-M3 ONNX 编码器（2026-10-27，未部署）

嵌入执行器单独配置：`HOJ_RAG_EMBED_ENGINE` 默认 `pytorch`；显式选择 `onnx` 还须配置 `HOJ_RAG_EMBED_GRAPH_PATH`，并保持 `HOJ_RAG_DEVICE=cpu`。原 `HOJ_RAG_MODEL_PATH` 提供固定版本 tokenizer / 配置，图路径同目录须包含 `model.onnx_data`。`onnx_embedder.py` 在启动时分别完整校验小图、外部权重及八份原配置的代码固定 SHA256，再创建 CPU-only FP32 会话；错误即阻止启动，无自动下载或原生回退。

该执行器使用原默认分词配置、8192 截断上限及空 query/document prompt，每批最多8条，恢复调用方顺序，复用原共享 inference gate。工厂查询预热完成后才开放服务；关闭或初始化失败释放共享会话，close 等待当前编码完成。现有重排执行器、上下文投影和索引版本配置保持各自职责。图目录必须作为独立只读资产挂载，不加入 Git 或应用镜像。

当前只完成组件与封装验证，尚无该编码器的完整300问、工厂/TCP生命周期或业务时延资格。没有新增 Compose 挂载或启用开关，本地运行服务仍为原 c151。具体证据与剩余工作以[质量检查点](hoj-evaluation.md)为准。

```powershell
docker compose --env-file HOJ-master/.env.rag.local -p hoj-master `
  -f HOJ-master/compose.yaml -f HOJ-master/compose.dev.yaml -f HOJ-master/compose.rag.yaml `
  -f HOJ-master/compose.video.yaml -f HOJ-master/compose.rag-rerank.yaml `
  -f HOJ-master/compose.rag-context.yaml up -d --no-deps --no-build --pull never hoj-rag
```

## 内部契约 v1

所有调用均为 POST，无查询参数；ingest / retrieve / delete 为 application/json，extract-pdf 为原始 application/pdf。每次请求使用新的 `X-HOJ-Request-Id`（32 位随机小写 hex）、`X-HOJ-Deadline`（13 位绝对 UTC 毫秒，最长 120 秒）和 `X-HOJ-Signature`（HMAC-SHA256 小写 hex）。密钥为共享凭据 hex 解码后的 32 字节。

签名原文为以下六项用 LF 连接，不添加结尾换行：

```text
v1
POST
/internal/v1/具体操作
请求ID
截止时间
原始HTTP请求体的SHA256小写hex
```

| 路径 | 请求 | 返回 |
| --- | --- | --- |
| `/internal/v1/ingest` | `documentId,version,contentHash,title,format,content` | `documentId,version,contentHash,state=INDEXED,chunks` |
| `/internal/v1/retrieve` | `query,allowedVersions,topK` | `{items:[...]}` |
| `/internal/v1/delete` | `documentId,version,contentHash` | `documentId,version,state=DELETED` |

`allowedVersions` 是 HOJ 根据当前权限与发布状态生成的三元组列表；每项仅含 `documentId,version,contentHash`。documentId 为 32 位小写 hex，version 为正整数，contentHash 为所传 UTF-8 content 的 SHA-256，不在服务内归一化换行。即使正文 / 文件名相同，不同资料和版本也不共享向量身份。

title 1–200 字，普通正文最多 200,000 字，PDF 正文最多 300,000 字；format 为 `markdown`、`text`、`pdf` 或 `video`。query 1–2,000 字，topK 1–10，允许版本最多 500 个且不重复。长度按 Unicode 码点计算；普通 JSON 请求最多 1 MiB，ingest 最多 4 MiB，原始 PDF 与其特殊期限见下文。多余字段被拒绝，不接受 UID、浏览器 Cookie、任意文件系统路径或模型密钥；视频只通过服务端资产标识读取约定目录。错误仅返回固定 code，不回显输入或异常正文。

items 包括 `documentId,version,contentHash,chunkId,chunkHash,chunkIndex,title,excerpt,headingPath,score`。score 是 cosine 相似度，不表示正确率或能力；当前没有相关性阈值或自动拒答。Markdown 仅使用实际 ATX 标题构建章节，忽略 front matter 的身份 / 权限 / 标题，代码围栏不变成章节；文本格式保持字面正文。服务不生成浏览器引用链接，HOJ 必须重新核对当前可见版本后自行构造链接。

同一版本重试保持幂等；正文 / 标题 / 格式变化需新版本。新版本失败不影响旧版；部分写入不进入 READY。删除先写 DELETED，再幂等清理对应版本点位，迟到重试不可复活。签名防重放与入库幂等独立，重试 HTTP 必须换新请求 ID。授权范围为空或没有 READY 的对应版本时直接返回空 items，不触发 Embedding 或无过滤查询。

## 验证

2026-10-23当前源码完整主机536项通过，正常Dockerfile/51源码比对、七组284问原质量/耗时门槛、隔离Linux实际工厂/TCP及CE删除并发、部署后真实HOJ/Vue验收通过；本地RAG为82aed8351e9c…。主机ASGI仍使用测试进程内存事件循环，不能替代单列的Linux/TCP证据；此前整本/视频证据保留，真实本人模型、生成/拒答和未见课堂资料质量未验收。完整口径、首轮耗时失败和教材缓存环境限制以[质量检查点](hoj-evaluation.md)为准，后面旧计数为历史。

```powershell
python -m pytest tests
python scripts/verify_hoj_bridge.py --model 'D:\完整本地BGE-M3模型目录' --hoj-root 'D:\IdeaProjects\HOJ-master' --log 'D:\临时证据目录\java-bridge.log'
```

Python 测试使用受控向量和真实 Qdrant 本地模式，验证权限范围与版本行为，不代表语义质量。第二条显式脚本使用真实本地 BGE-M3、独立临时磁盘索引和随机临时服务凭据，通过 Maven 调用 Java 客户端；校验实际 XML testcase 数，随后停止临时服务并清理。不会连接既有 HOJ 数据库或聊天模型。需要主机已缓存 Maven 依赖；测试覆盖 Lombok 版本仅用于主机 JDK，不能代替生产 Java 8 构建。

2026-10-19 全量 Python 实际 282 项通过（原有 240 + 适配器 42），有一条 Starlette 测试客户端弃用提示；未将构建成功代替测试计数。原 Java 独立桥接的真实 BGE-M3 联调证据保留；当前 HOJ 的 tests/knowledge-rag-local-integration.cjs 又完成真实上传 / worker / 服务 / 发布 / Vue 检索原文 4 组及 2 张截图，未用数据库伪设 INDEXED，额外验证坏新版本不影响旧发布和重启后无需重新入库。临时业务数据与向量清理，manifest 撤销墓碑保留以阻止迟到复活。当前完整证据以 HOJ docs/docs/ai/TODO.md 为准，两个仓库均未提交 Git。

资料仅由 root / admin 维护；用户未勾选时只在站内检索与查看引用，明确勾选后才允许 HOJ 将命中且仍可见的片段发给本人模型。HOJ 业务与页面已落实这些规则并通过受控模型 / 权限回归，真实本人模型尚未调用。固定题集仍有未完整进入前三的证据，无依据拒答 / 逐结论引用质量、PDF 公式完整性和未见课堂视频准确率尚未验收；实际样例通过不代表这些质量条件已满足。

## PDF 提取与页引用

`POST /internal/v1/extract-pdf` 仅接受签名原始 application/pdf，最多 32 MiB；签名同样覆盖精确原始字节。返回 sourceHash、pages、content、emptyPages、parser 五个字段，不创建索引、不接收文件路径 / UID / 用户 Key。JSON 入库最多 4 MiB，覆盖 Java 对补充平面字符的转义；PDF 提取成功响应最多 2 MiB，其他 JSON 操作仍最多 1 MiB。PDF 正文最多 30 万 Unicode 码点，Markdown / TXT / 视频仍最多 20 万；Java 严格核对原始哈希及每页 / 空页元数据。

提取优先使用固定 pypdf 6.21.1，遇到非法字体映射时，复用原 RAG 的 PyMuPDF（服务固定 1.28.0）重新提取完整物理页；回执 parser 只允许 pypdf-6.21.1 / pymupdf-text-v1。兼容路径仍先检查加密、页数和全部单页解压流，不跳过校验，也不删除非法字符强行成功。所有提取均在同一独立进程内，只继承运行所需 PATH / PYTHONPATH / 临时目录变量，不继承内部服务身份或用户配置。父进程最多等待 25 秒并杀死 / 回收超时进程，Linux 子进程仍限制 512 MiB 虚拟内存、20 秒 CPU。最多 200 页、30 万码点、单页解压内容流 4 MiB；空白页保留物理页号。文字提取不等于公式识别，已有教材仍有公式缺失，发布前必须对照原件核对；PyMuPDF 使用 [官方 Page.get_text 接口](https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_text)，不把返回无非法字符当作内容完整证明。

普通服务请求仍限制截止时间在未来两分钟内。只有完整 HMAC 验证后的 ingest 请求、且签名正文 format=pdf 时，才允许最多 20 分钟；其他格式 / 操作不能复用。HOJ PDF 后台入库租约为 21 分钟，其他任务保持 3 分钟，并保留旧 token / 过期完成隔离。浏览器上传只等待文字提取与持久化，不等待向量化；后台 INDEXED 后管理员显式发布。已有 HOJ 数据库先执行可重复的 sqlAndsetting/knowledge-pdf-capacity.sql，将 source_data 从 mediumblob 无损扩为 longblob；新库直接使用更新后的 knowledge-base.sql。本地原件下载并发降为 1，保持单批最多 32 MiB 原件的资源边界，普通资料阅读不受这个下载槽影响。

canonical content 用 `\n\f\n` 分隔每个物理页。ingest.format=pdf 对每页独立切片，headingPath 为“第 N 页”，PDF 内的 # 不解释为 Markdown。HOJ 保存原件及正文为不可变版本，校验片段确实出自对应物理页后才给用户 page 引用。原件下载 / 发布 / 权限以 HOJ 为准，Python 没有公开文件路由。

最终 Python 307 项、Java 50 项（含真实 MySQL RR）与 Linux 中文提取 / 源码哈希已验证；部署、真实全流程、页面及用户验收状态只在 HOJ docs/docs/ai/TODO.md 维护。测试 fixture 使用本机 CJK 字体生成并子集化的原创样例，不把字体或 PDF 文件提交仓库。

## 明确代码标识符补召回（2026-10-23，已部署）

默认 dense 路径新增有限补召回：问题明确写出 ASCII 下划线标识符，且语义 topK 的片段正文完全没有该名称时，才查询现有授权 FTS。单次至多处理四个、每个至多 64 字符的名称；普通自然语言、已包含名称和没有匹配来源时保持原响应。FTS 只查缺失名称，完整允许版本 / READY / namespace 在 LIMIT 前过滤；返回后从 Qdrant 读取真实向量与正文，按原始大小写和 ASCII token 边界核对，标题命中不算正文证据。

最多替换最后一条，前缀顺序与内容保留；若会丢失另一个已出现名称的唯一来源，则保留原结果。公开 score 仍是实际 cosine，返回前仍复核持久撤销状态。切片、向量输入、embeddingRevision=4 和 HTTP 契约不变，不需要重编码；旧资料缺关键词时仍可检索，管理员显式重建可沿用既有逻辑补齐关键词。普通 hybrid 实验仍按下节独立记录。该实现已随正常源码镜像部署；真实整本管理页上传、四问 / 物理页引用的最终报告及早期失败见 hoj-evaluation.md，不再沿用“Docker 尚未执行”的旧阶段状态。

## 入库与检索并发（2026-10-23，已部署）

服务仍是单进程、单模型实例，Docker 启动的 workers=1 不变。同一 client 的 KnowledgeIndex 实例共用存储锁和串行入库通道，同一 embedder 的实例共用推理调度。存储锁不覆盖模型计算；文档按原切片顺序每批最多 8 段，查询可在批次之间执行，但不能抢占正在编码的批次。查询优先，同时有文档等待时最多连续四次查询；所有队列 / 锁等待沿用请求截止时间。HTTP 仍最多两个操作占槽，入库另外限制单个，先取得入库槽才占共享槽，第二个入库直接 SERVICE_BUSY / 429，避免两个长入库排除全部查询。PDF / 视频提取配额仍独立。

检索编码前只快照当时 READY 的完整允许元组，编码后与最终返回前重新检查持久状态。入库在每批排队前、实际编码前和最终发布前检查不可变元组 / WRITING；删除先提交墓碑，不等待推理通道，旧作业不能继续下一批或发布。切片内容 / 身份、revision=4、公共请求与返回保持，已有 READY 索引不因更新自动重建。批次边界改变了大文档一次 encode 的分组，实际模型数值 / 固定问题验证单独记录；不声称任意大文档向量逐位相同。

当前完整主机单元 / ASGI 契约 507 项通过，生产启动没有调整。实际离线 BGE-M3 对原书重新编码 367 段，四问完整证据前五均命中，入库期间独立已发布资料查询约 3.6 秒返回；这是独立主机证据。自动审批恢复后，正常镜像 / Linux 并发与撤销检查已通过并部署，真实管理端整本上传期间的普通用户查询也已通过实际后端返回，Python 完整版本前后均 WRITING，MySQL job 仍 RUNNING；实际计时与完整业务结果见 hoj-evaluation.md。单次完成不能作为任意负载下 25 秒响应承诺。

## 实验关键词排序（未启用）

源码新增 SQLite FTS5 + BM25 与 dense 的独立候选合并，使用现有 state.sqlite3，不增加 HOJ SQL；关键词与 READY 回执同事务、撤销时先清理关键词，完整授权三元组和命名空间在截断前筛选。旧向量可用时显式重建仅补关键词，代码默认 retrieval mode 为 dense，只有运营方显式设 HOJ_RAG_RETRIEVAL_MODE=hybrid 才选择实验排序，用户不能传该开关。

目前新数据提高 top3，但旧数据 top1 / MRR 退步，未达到启用门槛，后续 OCR 镜像已包含投影源码，但 Compose 明确固定 dense；不要把源码部署当作混合排序已启用或已达标。hoj-rag:hybrid-eval-20261019 保留历史实验镜像；当前 local 镜像包含 OCR 与投影源码，实际排序仍固定 dense。方法、冻结新数据和全部结果见 [质量检查点的混合实验](hoj-evaluation.md#关键词--向量实验未通过默认启用门槛)，当前源码待后续重排验证，不是生成正确性或拒答能力完成声明。

## 扫描 PDF OCR（2026-10-20）

POST /internal/v1/extract-pdf-ocr 使用原始 application/pdf，HMAC 同样绑定路径、原始字节和截止；nonce 与正常提取共用，不能将其签名跨路径使用。共享单个 PDF 容量槽，普通提取行为保留。返回原 sourceHash / pages / content / emptyPages 及 parser=pdfium-tesseract-chi_sim-eng-v1；失败只返回固定错误码，不回显片段、密码或文件路径。PDFium 5.14.0 / Pillow 13.0.0 与镜像内 Tesseract chi_sim+eng 完成全页识别，混排页不会因页眉已有文字而跳过图片正文。

2026-10-22 起共享原始 32 MiB、200 物理页、30 万提取码点上限，OCR 的计算资源与时间预算保持原值。每页按 200 DPI 渲染前限制 1600 万像素、单边 1 万像素。Linux 子进程 768 MiB 虚拟内存、90 秒单进程 CPU、64 MiB 单文件；每页 Tesseract 20 秒，整体处理 80 秒，父进程最多 85 秒并杀死整个进程组。子进程仅继承运行必需环境，不接收身份、模型路径、用户 Key。超限整体失败，不跳过失败页后返回不完整成功；200 页是页数上限，不是处理时长承诺。

原始识别可能误读中文 / 公式 / 代码，实际样例误识句保留在证据。HOJ 允许管理员显式按页校订为新版本，保留原件 / 原提取 / 旧发布，以 OCR_REVIEWED 标注；索引仍读取 HOJ 发送的完整规范分页文本，不接收原件下载路径。原生文字 PDF 也支持同一物理页校订入口，标注为 NATIVE_REVIEWED；OCR 校订仍为 OCR_REVIEWED，原件、旧正文和旧发布保留，索引后须显式发布。该来源模式使用已有 varchar(16)，无需新 SQL。公开校订标记与重放契约见 HOJ docs/docs/ai/08-data-model-and-api.md，RAG 仍只接收规范分页正文。校订和发布权限仍由 HOJ 控制；当前部署、验证及完整目标状态见 HOJ TODO。

## 视频内容身份（2026-10-23，已部署）

视频 probe / transcribe 仅接受服务端资产 ID 和预先绑定的长度 / SHA，不接收浏览器路径。每次处理前后都校验源文件完整内容、regular file、路径 / 描述符及 dev / inode / size / mtime_ns / ctime_ns；按已声明长度读取、受截止约束，交给 worker 前恢复描述符偏移。模型内容版本也在转写前后重算，不再凭元数据缓存跳过。该修复针对真实 Linux 上同长度改写且五字段碰撞的残留内容变化，不承诺任意中途改写再完全恢复的不可变快照；资产保存仍须遵守只写一次规则。

真实 Linux 碰撞、模型修订及 FD 偏移检查通过，实际 80 秒原创中英视频的 ASR / 引用 / 播放 / 带向量重启也通过。额外完整读取的成本只测过 16 MiB 热 tmpfs 合成样例及该小视频业务流程；冷盘 / 1 GiB 媒体成本、未见课堂识别准确率仍未验证。完整证据与原失败记录见 hoj-evaluation.md。
