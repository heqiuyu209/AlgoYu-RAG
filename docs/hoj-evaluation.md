# HOJ 知识库检索质量

追加多证据覆盖诊断（2026-10-26，候选均未合格）：继续用户确定的“保持现有速度、改善证据完整性”方向。先在冻结V9的原始候选池上验证集合边际覆盖：固定现top2，以同模型ColBERT逐查询token的新增覆盖选第三条，保留来源存在性、代码标识和完整身份。实际19个query各一次编码，文档14切片/4个batch及1次warmup；前三由19/24提高到21/24、留出由7/11到8/11，前五22/24，无部署旧前三损失、各split聚合非退步，但原all/heldout的90%门槛仍失败。单片pointwise对照仅18/24，另丢c001旧前三，不能采用。5项focused检查通过；33份FP32矩阵/3792 token行已保存，独立审计重算245候选/8617项maxima和全部选择/指标通过。入口data/hoj-set-coverage-20261026/report.json；这不是新ANN、业务p95、生命周期或旧300题回归，未移植/部署。

后续位置权重假设已单次执行完：固定同一BGE revision的官方sparse_linear.pt（3516字节、SHA45c93804…），作用于同次query forward的raw hidden，ReLU后清零cls/eos/pad/unk；按位置加权新增覆盖，未使用官方词ID最大值聚合，不能称官方sparse检索复现。仅19次新query forward＋1warmup，文档编码0、复用前述14矩阵；保存完整CLS hidden/IDs/attention、dense、投影token及原始/掩码权重，共152数组。主候选前三20/24、留出8/11、前五21/24，质量失败；fresh unweighted与去特殊位置uniform均21/24，完整输出相同且重现首轮。weighted pointwise仍18/24并丢c001前三。主候选无部署旧前三损失、聚合不退步，却丢掉unweighted的c004/h005修复，h007原第五条也未返回，不能把局部修复称普遍改进。入口data/hoj-sparse-weight-coverage-20261026/report.json；独立完整数值审计已完成，88项hash/元数据及6测试核对通过，hidden重算dense/ColBERT/sparse最大误差约1.49e-8/1.04e-7/2.09e-7，所有选择与指标一致，candidateQualified=false。

权威封存在data/hoj-sparse-weight-coverage-20261026/final-state-v2.json，review-notes.md整理两位只读复核者的结论。初版状态把首轮33个总矩阵误标为query矩阵；v2明确为19 query＋14 document、共33，旧已hash快照不覆盖。独立来源调查另确认：本轮集合目标对应[DISCo，ICLR2026 §3.1公式2/附录F](https://proceedings.iclr.cc/paper_files/paper/2026/file/6ded182245f8e4db3963d2027174133d-Paper-Conference.pdf)的query-token集合覆盖，[官方贪心实现](https://github.com/structlearning/DISCo/blob/main/src/greedymethods.py#L65-L97)同样逐token取最大值。固定前二是本项目适配；公式并未验证事实关系或否定，覆盖分提高不等同必要证据完整，这是从公式得出的限制。没有因论文存在就宣称本地资格通过。

逐token贡献的独立只读核对已完成：h002压住泛词，口袋/配置贡献使正确BFS2进入第三；但h005“模型”误对齐背景model checking，占错误winner贡献95.09%，h007的bound误对齐Frontier，c004“数”的有效增量反被零权重删去。问题还包括关系与子词对齐，不能只按停用词修补；不引入题目/文档ID或人工词表规则，也不扫参数追已暴露留出集。unweighted评分FP32、uniform/learned变FP64，二者差异不能只解释为特殊token；本轮三者完整输出恰好一致的只是unweighted与uniform。

运行边界：本阶段11:49UTC检查发现原七容器约09:00UTC均已退出，OOMKilled=false、原因未确认，诊断尚未启动；只start原容器恢复，不归因新方法。12:26:30UTC新只读核对仍为原c1517e20…/后端/判题三镜像、55源码匹配、7运行6healthy、两API200、原计数3/8/9/6/0/1/0/0、raw/context各0，两个实验容器均移除；Git双HEAD/分支保持，未SQL迁移/删卷/Git提交。data/hoj-sparse-weight-coverage-20261026/runtime-final.json记录此时点。当前线上版本保持；以下原300题部署证据仍只覆盖旧范围，课堂质量及完整七项目标继续未完成。

追加真实课堂验收（2026-10-26，质量未通过，当前部署保持）：新增MIT OpenCourseWare 6.006 Fall2011的BFS/DFS两份官方PDF，首次用于本项目；保留原件、出处、CC BY-NC-SA4.0署名及原生解析文本，隔离评测而非自动导入站内资料。独立source-only作者先写20题、首测前调整来源范围并补两道跨讲义题，再冻结10文件；12中/8英，两个split各8正2负，24组必要证据。作者未读实现/排名，最终抄录和165/116词引用另核对通过；不是学生自然查询或专家验收。两份新文档的首测不能称模型预训练未见资料。

追加结果已独立复核：三模式的质量完全相同，前三19/24（79.17%）、前五21/24（87.5%），留出前三7/11（63.64%），16可答题中12题前三证据完整；原90%质量门槛失败，逐题前三保护及all/calibration/heldout聚合非退步通过。这改变了“只剩旧文章一处遗漏”的下一步判断：课堂讲义仍有4问/5组证据未进前三，均已进入raw候选池。出处与冻结输入在evaluation/hoj-classroom-v9；权威封存在data/hoj-classroom-eval-20261026/final-state.json，原始合并审计保留combined-audit.json。本轮没有业务源码或运行镜像变动，也不重跑不变模型直到变绿。

| 新讲义漏项 | 必要物理页 | raw池排名 | 实际现有评分 | 返回排名 |
| --- | --- | ---: | --- | --- |
| c004，BFS扫描复杂度 | BFS6 | 7 | 未评分 | 前五未返回 |
| h002，魔方配置图 | BFS2 | 11 | 未评分 | 前五未返回 |
| h005，调度起点 | DFS5 | 4 | 已评分 | 4 |
| h007，跨讲义复杂度 | DFS2／DFS3 | 7／5 | 未评分／已评分 | 未返回／5 |

评测返工和范围：初次driver在第4题沿用旧集的top2完整对象逐位相等断言中断，前三题9响应保留；单例诊断补c004的3响应，剩余16题另进程48响应，共20唯一题/60保留响应、各模式19次query编码。首轮c004的3份未保存返回不纳入，原失败source/report/log保留；不是单进程20题计时。后三个阶段不混算正式p95；续测16题同轮candidate476.23ms、预算552.24ms仅说明该阶段耗时通过，不能覆盖完整20题或质量失败。285条返回/490条pool引用核对，240条分数对当时实际raw向量重算；初三题45条未保存query向量，不冒充重算。

数值原因已闭合：Qdrant Local1.18.0的search把self.vectors基本切片view传给cosine_similarity，后者vectors/=norms原位归一化内存矩阵；retrieve仅.tolist，无查询路径持久化写入。单例三模式1024维query列表逐项相同，top2身份/原文/引用相同，仅score有6.499e-8差；续测再有一处7.336e-8。主机三个SDK源码匹配wheel RECORD并与实际镜像hash逐字节匹配，不归因limit5/20。后续分别验证前缀payload/identity严格相等及实际raw余弦误差≤既有1e-6，精确浮点对象差单独记录；没有改变题目、标签、90%或时延门槛，也没有改业务选择器。

追加真实分类器诊断已完成：对已暴露4漏题的14候选全池各评分一次，56对/28个batch2，无新query编码/ANN。c004必要片CE第3、h002第3、h005第5，h007两必要片第2/8；有些正确片未进入现评分预算，有些即使评分也不在最佳三条，多证据覆盖不能靠单纯扩大CE保证。直接分类器约544–687ms不含BGE/ANN，不能当业务p95或合格方案；不发布全池规则。入口data/hoj-classroom-ce-diagnostic-20261026/report.json。其inputAuditSha指向运行时中间审计；随后仅为清理审计脚本写法重算，覆盖了该中间JSON的时间戳而未留旧副本，此hash链缺口保留，不声称与当前audit文件hash一致；直接三份权威report、冻结源、实际scoredPool和模型身份可独立核对。

独立复核边界：240条原向量点积由实际collector当场计算，独立代理核对记录与源码；未保留原向量本体，不能称其独立重新计算点积。实分反事实“固定现top2＋全CE尾部”仅21/24前三，h005/h007还各失去一条原前五依据；固定top2使h007最多2/3组进前三，但全题oracle上界仍23/24、heldout10/11，不能据此声称过90%必然要放松top2。07:09:57UTC只读封存仍为c151/原后端判题、55源码、7运行6healthy、API200、原业务计数、raw/context各0，所有实验容器移除。

版面质量边界另列：六项观察涉及代码缩进、求和/分段公式次序、图的箭头/顶点和完成时间对应丢失；抽取文本可读及检索命中不表示这些信息已恢复。原PDF可供管理员校订，未用修改标签绕过抽取问题。后续审计重跑写带UTC的新文件，已被下游hash引用的快照保持不覆盖。当前原300检索回归/部署资格仍只覆盖原资料；真实课堂质量和完整七项目标未完成。下一步验证通用的多证据候选覆盖与选择，并保持原质量、速度、权限与引用保护。

当前实现进度（2026-10-26，集成300问与独立复核通过，已部署本地）：冻结选择器/双表示检索已移至业务包passage_selection.py、dual_index.py、paired_index.py，单索引ingest/delete抽成内部helper。共享lane只持一次，raw READY仍补建缺失context；每批编码及最终发布复查raw权威状态，双READY后返回INDEXED；仅canonical chunk和完整输入一致、实际raw payload/revision/vector核验后复用。删除先双tuple预检、双tombstone提交，再物理清理，失败可幂等重试。service.from_env现已显式接入固定ONNX执行器和独立context-r1；默认旧路径不变，context严格要求dense＋ONNX，配置错误或预热失败拒绝启动。

本轮新证据：factory13＋ONNX15＋旧reranker18共46项通过，完整tests/hoj为349通过、0failure/error/skip。常规Dockerfile构建候选sha256:c1517e20ee0d003afa71d117e32d1874bcfeb6f8762d1eaa89f974e34325beac；固定Transformers5.20.2/ORT1.31.0，镜像全量Python与仓库源码hash一致，data原型未进入运行链。真实本地BGE/固定图经from_env创建实际应用、断网loopback TCP验证六项检查通过：签名/错签/重放/截止/空权限、相同输入向量复用与幂等、真实tokenizer512评分和inactive513整组回退、评分gate占用时提交双tombstone并拒绝迟到结果、正常lifespan关闭。人工暂停在realORT调用之前，不称原生kernel中断。首次探针把空文本特殊token数用于非空文本，断言失败，source/report/log完整保留attempt-1-tokenizer-fixture；仅改实际token计数构造后合同探针exit0，未修改业务代码或计时结果。入口data/hoj-context-factory-live-20261026/report.json。新增第六份Compose配置及七挂载检查通过，现已应用。

唯一新集成进程已exit0、容器移除：data/hoj-context-integrated-20261026实际八组300问/900响应，每模式298次query编码，两空权限零模型；八组原全部质量/逐题/聚合/耗时及额外旧FP32保护通过，不沿用原型计时、不删除冷观察。实际paired导入编码209份完整输入（含不同context输入的部分重编码），两份manifest均READY；主审计重算4329返回/10206pool/274评分/1226active、349 XML、55源码hash、77freeze及引用/权限，完整返回与旧冻结消融300/300一致。公开仍17/18、heldout10/10、旧h008[1,3]；v8 dense248.69/current574.47/candidate487.84ms，原预算498.69ms，余量10.85ms，stableLatencyEstablished=false。唯一新进程467.06秒，原120000ms请求截止/1800秒watchdog不变。

| 业务集成组 | candidate p95 / 同轮原预算 | 结果 |
| --- | ---: | --- |
| v1 | 422.14 / 515.30ms | 原全部门槛通过 |
| v2 | 376.65 / 477.06ms | 原全部门槛通过 |
| v3 | 403.72 / 477.44ms | 原全部门槛通过 |
| v4 | 492.18 / 559.47ms | 原全部门槛通过 |
| v5 | 493.26 / 551.60ms | 原全部门槛通过 |
| v6 | 566.95 / 636.35ms | 原全部门槛通过 |
| v7 | 533.23 / 636.11ms | 原全部门槛通过 |
| v8 | 487.84 / 498.69ms | 原全部门槛通过，余量小 |

运行时点与复核边界：04:12UTC旧服务健康快照后，06:14UTC重新检查发现原七容器均已退出（RAG在04:34:46UTC，OOMKilled=false），原因未确认，候选当时未部署；只start原容器恢复，06:17:53UTC的新runtime-recovered.json确认原三镜像、7运行/6healthy、两API200、计数3/8/9/6/0/1/0/0、零向量及原双HEAD。不将04:12快照冒充当前状态。独立审计首次因代理平台使用额度中止，未记为通过；过工具提示恢复时点后由同一代理完成全量复核，1200指标对象/900历史基线/300缓存及原型输出完全核对。复用向量与旧独立编码的241场景/267个内部contextCosine有浮点差异、最大3.1693741e-7，池顺序/公开raw score/完整返回不变；dense两处数值路径差仍保留，不宣称所有内部值逐位等同。

本地部署已完成：06:24:16UTC deployment.json记录进程exit0、无rollback，仅RAG切换至c1517e20…；55个运行源码匹配合格镜像，原后端1c5b6027…与判题ac1ae3d7…不变，7运行/6healthy、两API200、原计数不变、raw/context各0向量、服务身份相同。旧五覆盖加context第六，七挂载及原数据/身份/媒体三命名卷保留，图/分词器/身份均只读，8097未公开；06:25 live-config-check另从backend容器访问受保护路由返回匿名401，验证内部网络可达，不冒充真实用户知识问答。图路径只写入已忽略.env.rag.local，无SQL迁移/删卷/Git提交。部署脚本先恢复.env再rollback Compose，状态写入成功后才记deployed=true；此失败路径只经静态复核，未故障注入。权威入口data/hoj-context-integrated-20261026/final-state.json及deployment.json，candidateDeployed=true、independentReviewComplete=true、goalComplete=false。

下一步仍保留完整七项目标：公开长文章还有1/18必要证据遗漏，速度余量小、未证明稳定负载；真实生成/拒答、未见课堂质量及用户延后的本人模型/源站AC-WA仍未完成。不能将本轮检索与本地部署通过称为整套产品完成。

先前生命周期检查点（集成前历史）：原选择器函数/Context/readiness/retrieve和单索引ingest/delete默认路径经独立AST核对等价；业务新类实际方法以冻结logit纯内存重放300/300一致，另73份文档/188切片的embedding输入与payload helper等价，不冒充新模型回归。生命周期首轮15项加旧projection/runtime共36项通过，补正常query原cosine/整池模拟回退后，完整tests/hoj为321通过（paired17、其余304），无failure/error/skip，独立审查及XML核对完成。HTTP签名/权限/隔离检查使用真实本地SQLite/Qdrant及ASGI、模型均为fixture；正常query从实际存储向量重算cosine容差1e-6，回退只模拟score返回None，不称真实tokenizer512/513验证。权威入口data/hoj-paired-lifecycle-20261026/final-state.json：lifecycleHostContractQualified=true、productionFactoryWired=false、actualNewRealModelRegressionExecuted=false、candidateDeployed=false、goalComplete=false。旧index精确快照保留pre-change/index.py并与10-25原300源码hash匹配；原型通过不等于修改后的集成版本已通过同一真实回归。

环境与恢复边界：第一次完整主机回归在沙箱的Windows asyncio本地socketpair初始化挂起，py-spy只读栈确认停在TestClient业务启动前；同Python沙箱外初始化成功，仅终止这轮已核对的两个测试PID并保留原日志/栈，再以相同代码和断言在允许loopback环境执行，321项通过。跨日Docker最初未运行，恢复Desktop遇到旧dockerInference AF_UNIX端点Error1920；正常停止失败后仅结束本次Docker进程，单节点Rename-Item仍1920，端点保留、未重置配置/卷/WSL。后来只读观察到新的Desktop/引擎已可用，恢复原因未确认，不将其归因于失败的改名。01:46UTC封存核对原三镜像、7运行/6healthy、API均200、业务计数3/8/9/6/0/1/0/0和零业务向量；不是新版本部署。data/hoj-primary-full300-20261025/final-state.json的currentRuntimeRechecked=false只描述先前离线审计时点，当前运行快照以本轮生命周期seal为准。

冻结原型历史（2026-10-25，当时全量门槛通过、生产接入未完成）：新的FP32候选保留全部primary incumbent和原secondary的m+1比较，仅省去额外primary challenger；其余全pool分词/512回退、旧selector复用、跨来源比较、代码保护、vector supplement和权限复查不变。300题真实分数缓存的一次消融全部原质量/逐题/聚合门槛通过，实际旧/新selector纯计算又各300/300复现完整对象；274个评分场景的pair由1242降至1226，16个mask变化，均为旧mask子集，旧challenger分数没有泄入新比较。只有v1两题完整返回变化、指标不退，公开仍17/18；缓存本身不是新300题模型回归。11项新类合同检查通过，覆盖单源绕过、legacy复用、全pool模拟超长回退、混来源模型期间raw/context撤销、并发/空权限等，不扩大成真实512/513 tokenizer边界证明。入口data/hoj-primary-incumbents-20261025/report.json及tests.xml，独立复核完成。

新40问真实阶段已完成并独立复核：v7 p95 521.21ms/同轮预算636.43ms；公开v8 p95 443.21ms/预算488.89ms，公开仍17/18、heldout10/10，旧h008保住[1,3]。两组原全部门槛及对上一FP32候选的额外逐题/各split非退步均通过；40次编码/模式、120响应、585条返回/1456条pool引用、29评分场景/126实际active逐项核对，与冻结消融完整返回无差异。入口data/hoj-primary-query-20261025/final-state.json，subsetOnly=true、candidateQualified=true仅是阶段资格，fullRegressionQualified=false、candidateDeployed=false。生成器第一次整段替换未匹配旧预检文本，独立检查在模型执行前发现fp32_reference未定义；已修正生成及实际driver并加入AST/赋值断言，未产生失败模型回归。

10-25原型完整实测（历史）通过原门槛：唯一新断网进程重新编码八组全部资料、实际300问/900响应，未复用40题计时；每模式298次真实query编码，两处显式空权限零编码/模型/候选。八组原90% all/heldout质量、dense/current逐题与各split聚合非退步、原max(dense×1.5,dense+250ms)耗时预算均通过，额外对上一FP32候选的逐题/聚合检查也通过。公开v8仍17/18、heldout10/10，旧h008保持[1,3]；实际完整返回与冻结缓存消融300/300一致。原dense/current完整对象复现历史，主代理及独立代理重算4329条返回/10206条pool引用、274评分场景/1226实际active及source hashes。权威入口data/hoj-primary-full300-20261025/final-state.json：validated=true、subsetOnly=false、candidateQualified=true、fullRegressionQualified=true、independentReviewComplete=true、candidateDeployed=false、goalComplete=false。v2–v8的77个freeze文件hash核验，v1无独立freeze，按现数据/切片和历史完整对象核对；dense补齐与池检索有两处cosine数值路径差异（最大约4.44e-8），不声称跨模式逐位同分。进程exit0、容器按--rm移除，总耗时508.69秒；单请求120000ms与job watchdog1800秒保持。

| 本轮primary-incumbent FP32全量组 | candidate p95 / 同轮原预算 | 结果 |
| --- | ---: | --- |
| v1 | 340.08 / 453.77ms | 全部门槛通过 |
| v2 | 331.46 / 443.67ms | 全部门槛通过，含空权限 |
| v3 | 334.77 / 441.89ms | 全部门槛通过 |
| v4 | 383.22 / 478.01ms | 全部门槛通过 |
| v5 | 392.95 / 475.23ms | 全部门槛通过 |
| v6 | 505.83 / 579.76ms | 全部门槛通过 |
| v7 | 530.47 / 615.58ms | 全部门槛通过，旧退步保住 |
| v8 | 545.99 / 552.14ms | 全部门槛通过，仅余6.15ms |

性能边界：公开集同轮dense302.14/current608.42/candidate545.99ms，预算仍按原公式计算，未改成前轮固定预算或删除冷观察；余量仅6.15ms，不声称稳定性能或绝对快于前轮491.66ms，40题443.21ms也不覆盖本轮实际值。不重复同一方案直到变绿。原型审计核对51运行源码、原三镜像及业务状态；该阶段生产仍a7a76b59…和15/18；当前部署以本页开头的业务集成检查点为准。生命周期模块与主机检查现已完成，见本页开头；下一步接实际factory/固定ONNX执行器与独立context-r1存储，验证实际工厂/权限/调用链，并对集成版本执行完整回归后再部署。完整七项目标保持进行中，检索全量通过不等于整套产品或RAG生成质量完成。

上一候选完整结论（2026-10-25，拒绝部署）：FP32 ORT batch2加版本状态批量读取已完成八组300问/900响应、同一新进程的真实回归，不复用阶段40问计时。全部题集的原质量、旧前三保护和all/calibration/heldout聚合非退步通过，旧v7-h008保住[1,3]，公开v8仍17/18必要证据前三；但v8 p95 491.66ms超过同轮预算460.90ms，整体拒绝。原dense/current完整对象逐题复现历史，两处空权限零编码/模型/候选，每模式298次真实查询编码。权威入口data/hoj-ort-full300-20261025/final-state.json：validated=true仅为审计完成，subsetOnly=false、candidateQualified=false、fullRegressionQualified=false、candidateDeployed=false、goalComplete=false。主代理重算4329条返回、10206条pool引用、274个评分场景及原门槛；受代理平台额度影响，独立复核未完成。此前40问484.23ms阶段通过只余2.75ms，不能覆盖这次全量失败，也不重复不变模型直到变绿。

| 上一FP32 ORT候选全量组 | candidate p95 / 原预算 | 结果 |
| --- | ---: | --- |
| v1 | 336.52 / 455.68ms | 原全部门槛通过 |
| v2 | 320.90 / 435.29ms | 原全部门槛通过 |
| v3 | 325.20 / 462.74ms | 原全部门槛通过 |
| v4 | 396.20 / 483.46ms | 原全部门槛通过 |
| v5 | 392.74 / 518.47ms | 原全部门槛通过 |
| v6 | 403.22 / 563.94ms | 原全部门槛通过 |
| v7 | 454.21 / 586.36ms | 原全部门槛通过，旧退步修复 |
| v8 | 491.66 / 460.90ms | 质量/非退步通过，耗时失败 |

接续诊断已完成并拒绝被动等待：data/hoj-thread-policy-20261025在两个新隔离进程中真实计算BGE query后接同FP32 ORT batch2，17缓存池各两重复，每策略34次真实query编码/34份选择，混来源各14次观察。默认/显式OMP_WAIT_POLICY=PASSIVE的模型交接段p95为499.76/696.22ms，进程CPU时间p95为1825.49/1942.21ms，显式被动等待更慢且未减少观察到的CPU成本；不采用该设置。两策略的新查询向量hash逐项一致，完整缓存返回与旧batch2一致，无旧前三损失；主代理重算通过，权威入口final-state.json，passiveWaitRejected=true、candidateDeployed=false。实测加载两个libgomp库、ATen OpenMP4线程和4线程MKL；这些事实不证明两个库本身导致性能失败。[GNU文档](https://gcc.gnu.org/onlinedocs/gcc-9.5.0/libgomp/GOMP_005fSPINCOUNT.html)和[LLVM文档](https://openmp.llvm.org/design/Runtimes.html)支持默认自旋/显式被动等待这一待测机制，并不承诺提速；本机结果排除此路径。

分阶段与图优化诊断已完成：data/hoj-stage-layout-20261025使用同17缓存池、两重复、两个独立FP32 ORT session，固定batch2/4线程/CPU、显式关闭GELU近似，仅比较BASIC/ALL。68次真实BGE查询编码及完整选择均核对，查询向量hash和旧batch2完整返回一致。混来源各14次观察的耦合p95为584.93/686.53ms，ALL没有收益，不采用。BASIC混来源阶段中位数分别为BGE 223.95ms、session.run合计240.15ms、完整pool分词9.38ms、pad 1.37ms、输入转换0.074ms、gate进入/退出及包装0.142ms；这些中位数不能相加冒充p95，gate项也不是纯排队时长。本轮没有并发查询或入库竞争；未发现值得优先优化数据转换的成本证据。

计时结束后另做两个未计时classifier调用并检查实际算子事件：主算子数量未显著改变，ALL仅有部分Gemm/Tanh融合为FusedGemm，观察到的样本无低精度类型/量化算子；这只覆盖一题实际执行事件，不是全图所有路径或唯一节点清单。额外profile中的Attention duration不能混入正式计时，也不足以归因。主代理和独立代理完成代码、分阶段指标、query hash、完整输出及原始profile hash/事件重算；入口data/hoj-stage-layout-20261025/final-state.json，diagnosticOnly=true、allOptimizationRejected=true、candidateDeployed=false。

ORT线程对照也已完成并结束该微调路径：data/hoj-ort-threads-20261025固定BASIC、FP32、batch2、inter-op1、SEQUENTIAL及关闭spinning/GELU近似，仅将ORT intra-op 4→2，BGE/OMP/MKL仍4。两session各复用17题两次、轮换顺序，共68次真实查询编码；各模式64个实际forward logit逐值相同、完整返回/active身份/forward shape/query hash一致。4/2线程混来源耦合p95为646.11/876.24ms；线程2在两次重复中的成对session.run中位数分别慢170.12/117.57ms、wall慢161.48/79.26ms，14组成对forward观察均更慢，不晋级真实检索，不再重复相同线程参数。独立只读复核一致，入口data/hoj-ort-threads-20261025/final-state.json，thread2PromotedToFreshQueryStage=false。

结构简化也被反例否决：预先固定“预算内至少两个primary完整身份在raw/context中全出现且同序，则省去可选primary/global-consensus评分、保留旧selector与vector supplement”的假设。两例缓存消融确认h002满足条件，但其secondary真实CE分数高于全部primary；省去比较让必要证据3→5，supplement补不回；c003仍缺，公开集上界16/18，不达原90%，立即拒绝，不为反例加条件。data/hoj-primary-elision-20261025/report.json只检查两个已暴露案例、复用已有返回/分数，无新编码/模型/ANN/计时或300问声明；独立代理确认反例。初次缓存记录把未触发分支也标为incumbentResponseReused=true，原脚本/报告/log保存为*-initial，仅修正该记录标记后重做同两例，选择和结论不变。

官方INT8图也已实际验证并拒绝：相同固定revision的[官方文件](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/blob/1427fd652930e4ba29e8149678df786c240d8825/onnx/model_qint8_avx512_vnni.onnx)118,620,017字节、SHA256为1825907d6c1a9001ff78124780bbde20a614a8c3df3b63409cf3c72c6fe5c8b4，按官方元数据下载校验到独立目录；实际CPU支持AVX512/VNNI，使用镜像已有ORT1.31。17缓存池两重复、FP32/INT8两个独立session固定BASIC/batch2/线程4，68次真实BGE编码；公开仍17/18，但旧h008两次均[1,3]→[1,5]，对current/同轮FP32逐题非退步失败。FP32/INT8耦合mixed p95为624.79/668.84ms，session.run合计中位239.84/293.94ms，两repeat paired forward分别慢48.70/42.12ms；即使进程CPU p95下降，也没有wall收益。64个真实logit最大差1.53412，336条返回引用/真实cosine与query hash经主/独立代理核对；入口data/hoj-ort-int8-20261025/final-state.json，qualityDiagnosticPassed=false、int8PromotedToFreshQueryStage=false，未晋级ANN或全300。它不同于旧Torch INT8实验，也未把原本漏掉的c003称为“恢复”。[ORT说明](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)明确量化可能损失精度，CPU指令与文件名都不能代替实际验证。

以上诊断ANN仍冻结，新查询向量未用于重新检索，不含ANN/manifest/完整业务耗时，不能用相减推算合格或跨报告归因提速。图/线程/INT8诊断进程均已终结、容器移除；正在运行的新预算全300是独立新job，见本页开头。原三镜像、51源码、七服务/六healthy及业务计数在40题审计再次核对，生产仍15/18、默认等待/线程策略未改，无SQL迁移/删卷/Git提交；不按题目/资料ID或经验分差补洞。

此前阶段：FP32 ORT batch2加版本状态批量读取通过40问/120响应的原全部质量、逐题/聚合非退步和耗时门槛。v7 candidate483.42ms/预算572.41ms；公开v8 candidate484.23ms/预算486.98ms，仅余约2.75ms。完整返回与上一批量2版逐题相同，旧v7-h008[1,3]保住。阶段入口data/hoj-manifest-query-20261025/final-state.json：subsetOnly=true、candidateQualified=true但fullRegressionQualified=false、candidateDeployed=false；不作全量、稳定性能或发布结论。

后续精度与模型验证（2026-10-25，未部署）：保持用户选择的现有速度和原质量/非退步门槛。精度分配已真实测完v7的24问和v8的16问，共40问/120响应、每模式40次查询编码；两组全部质量与非退步检查通过，但candidate p95分别634.93/599.84ms，超过同轮预算583.06/506.20ms，拒绝部署。原dense/current完整对象复现历史，585条返回引用、1456条pool引用及29个评分场景的互斥FP32/BF16掩码经主代理和独立代理复核。权威入口data/hoj-precision-query-20261025/final-state.json；subsetOnly=true、candidateQualified=false、fullRegressionQualified=false，不能称300问新回归通过。

新增5项边界测试实际零失败：adapter在FP32与BF16操作之间截止、两查询掩码隔离；FP32阻塞期间分别删除raw/context后最终409；模拟完整pool超长时只走旧FP32子集。最后一例不是实际tokenizer的512/513边界，deadline/并发例直接调用adapter，不扩大其覆盖声明。它们也不能覆盖原型尚未实现的双投影入库生命周期。

局部FP32诊断同样拒绝：data/hoj-fp32-head-20261025用公开16问和旧v7-h008的真实缓存、四模式各两次。只将classifier，或最后一个encoder layer加classifier保留FP32，其他算子BF16；代码复制模块映射、没有修改原权重/dtype或污染原模型图，经独立代理只读检查。两种局部FP32和全BF16仍使旧题[1,3]→[1,5]；全FP32为[1,3]。7个混来源案例每模式14个直接观察，FP32/BF16/head/head+last的p95为482.54/269.34/278.20/264.08ms；仅为模型/选择诊断，不含BGE/ANN/权限复查，不能据此定位全部误差或称业务达标。

英文MiniLM-L6对照也拒绝：从[官方模型](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)固定233902d25c440f23af6f7d6e94d2946bac0bee0a，七文件91,819,431字节按官方Git blob/LFS身份校验，safetensors SHA为821d1aa6…；原生BERT 6层/hidden384/512上限，加载无缺失或多余权重。六个已暴露公开诊断案例加v7-h008，有限预算+context及完整raw20共识+context各两次；保持原首两条、来源/原文/真实cosine，旧题恢复但公开c003/h002仍缺。即使未跑的十题全部命中，两方案整组最多16/18，低于原90%，不再扩大该模型回归；没有中文效果或业务耗时声明。直接段p95为275.62/1185.79ms，含首个新形状开销，未剔除冷观察。首轮评测字段漏补document而中断，失败源码/日志/报告保留，只修记录字段后重跑同组；Docker未启动造成的一次启动失败没有模型结果。权威入口data/hoj-minilm6-diagnostic-20261025/final-state.json，主代理重算164份选择/812条返回身份、指标及哈希；本轮MiniLM独立代理因平台使用额度中止，独立复核未完成，不是本地模型额度。

同模型FP32执行方式对照：从[官方固定版本](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/tree/1427fd652930e4ba29e8149678df786c240d8825/onnx)下载O2图470,605,122字节，按该revision的官方LFS SHA 04aca53e…校验；CPUExecutionProvider、4 intra-op/1 inter-op、sequential，未请求O3 GELU近似/量化/低精度。[官方优化说明](https://huggingface.co/docs/optimum-onnx/en/onnxruntime/usage_guides/optimization)区分O2融合和O3近似。原分词器、512整组回退、4对物理batch、真实候选mask/同源槽位保持。17缓存案例/两重复/两模式的68份完整返回和forward形状与原FP32一致，观察到logit最大差7.63e-6，不称数值等价；336条引用身份/原文/位置重新核对，混来源直接段p95原生862.87/ORT342.47ms，不含BGE等业务阶段，也不跨报告宣称稳定收益。

ORT真实40问仍拒绝：data/hoj-onnx-query-20261025重新编码两组资料和120份实际响应，每模式40次真实查询编码、一次查询向量共享。v7质量/非退步/耗时全部通过，candidate508.71ms、预算601.08ms；v8仍17/18且无旧前三/聚合退步，但candidate610.66ms超过585.33ms预算，整体验证失败。权威入口final-state.json，subsetOnly=true、candidateQualified=false、candidateDeployed=false。主代理重算585条返回/1456条pool引用和29个评分场景；独立代理受平台额度影响未完成，不作独立通过声明。审计首轮发现生成器按LF文本算hash、Windows落盘为CRLF，失败源码/日志保留；分别验证原始执行文件hash和仅换行规范化后的生成hash后补验，未重跑模型/改输出。审计生成器另一次字符串语法错误发生在生成audit前，失败源码亦保留。

ORT物理batch诊断：data/hoj-onnx-batch-20261025在同一FP32 ORT session上对同17池各两次，仅把生产score方法中的物理batch4改成2或1；未改tokenize/mask/长度排序/deadline/gate/选择器。全部完整输出跨batch一致，旧h008保住；7混源案例每模式14个直接观察，p95 batch4/2/1为351.53/284.55/301.29ms，保留所有冷形状观察。102份选择和504条返回经主代理核验，实际padding token cells分别20738/18906/17944；更少padding并不保证更快。批量2另轮真实40问/120响应仍拒绝：v7全通过，v8质量/非退步通过但502.88ms超过499.45ms预算约3.43ms；失败完整保留于data/hoj-onnx-batch2-query-20261025/final-state.json，没有重复不变方案直到通过。

状态读取优化：data/hoj-manifest-batch-20261025仅覆盖实验_ready，一次检查共用一个SQLite连接，按最多64个完整namespace/document/version/hash组合批量读READY，保留请求顺序、原查询前/推理后检查；没有跨请求缓存或放宽权限/撤销。1004项允许列表实测从1004次连接降为1次/16条有界SQL。原19项双表示检查和2项真实SQLite检查共21项通过，覆盖来源补取、原始cosine、跨源评分回退、并发隔离、模型期间raw/context撤销最终409、空权限、命名空间/版本/哈希和重复顺序。首轮因沙箱默认临时目录不可创建出现16项setup error，5项无需fixture者通过；失败XML/log保留，仅切换到已确认未使用的工作区basetemp后21项全通过，未改断言。

当前运行：Docker Desktop在接续期间退出，原六个HOJ容器处于停止状态；恢复Desktop并仅启动原容器，未重建或迁移。最新完整300问审计核对51源码、原三镜像、七服务/六healthy、API就绪、原业务计数与零向量，Git头不变，全量实验容器已清理；后续模型等待策略审计再次核对原镜像和七服务/六healthy，两个诊断容器均清理。没有SQL迁移/删卷/切分支/Git提交。所有模型实验断网且不接业务卷；只有实际质量/耗时和完整回归合格，才继续双投影生产生命周期。完整评测job上限1800秒，实际470.86秒；单次请求仍120000ms，质量/耗时gate未变。

完整回归结论（2026-10-25，拒绝部署）：用户选择的“保持现有速度、提高证据完整性”及原门槛保持。双表示/单源跳过CE/BF16候选已完成八组300问、900份有效响应的真实回归，整体不合格；在线仍为a7a76b59…、原15/18必要证据进前三。最新权威入口data/hoj-dual-regression-resume-20261025/final-state.json，validated=true仅为审计完成，fullRegressionQualified=false、candidateDeployed=false、goalComplete=false，不能用此前独立16问通过覆盖本次全量失败。

回归证据：首次300 driver在v2第64题的显式空权限样例上错误要求一次编码，服务正确返回空、零编码，评测中断；原probe/report/log保留于hoj-dual-regression-20261025。只修诊断计数/零评分/零observer分支后，按冻结顺序和模块哈希核验并复用有效v1的78题及v2前63题，共141题/423份有效响应；在新隔离进程真实执行剩余159题/477份，合并覆盖300题/900份，不称一次进程新跑900份。两处空权限每模式0编码，合并每模式298次真实query编码。独立代理和主代理重算全部指标，4329条返回引用、10206条pool引用、来源与cosine、模块/输入哈希均核对，原dense/current逐题复现发布历史。

| 题集 | candidate p95 / 原预算 | 结果 |
| --- | ---: | --- |
| v1 | 373.05 / 459.52ms | 原全部门槛通过 |
| v2 | 311.65 / 442.17ms | 原全部门槛通过 |
| v3 | 329.80 / 441.71ms | 原全部门槛通过 |
| v4 | 370.14 / 478.90ms | 原全部门槛通过 |
| v5 | 370.63 / 477.29ms | 原全部门槛通过 |
| v6 | 425.98 / 555.60ms | 原全部门槛通过 |
| v7 | 452.62 / 573.80ms | 旧前三/heldout召回/current聚合失败 |
| v8 | 508.16 / 495.36ms | 17/18但耗时失败 |

唯一旧前三损失为v7-h008：SAM/回滚并查集的两组证据由current的[1,3]变为[1,5]，heldout必要证据召回降至9/11。supplement前后ID完全一致，问题发生在评分选择阶段。两个rollback片段的真实BF16 logit为-0.9609375/-0.96484375，并非平局；旧完整FP32缓存为-0.9892546/-0.9715411，排序相反。不能用PowerShell显示的两位小数推断精确相等，也不能用“只重算精确平局”声称修复该例。v8仍17/18，但同轮dense245.36/current532.01/candidate508.16ms，原预算495.36ms；旧current也超同轮预算是观察事实，不改变候选gate，不重跑直到变绿。

后续诊断：data/hoj-limited-fp32-batch-20261025对真实公开16问加旧退步题共17问、两重复，FP32 batch4/2/1的直接选择段p95为473.51/477.59/401.90ms，完整输出跨batch一致，旧退步题均恢复[1,3]；实际forward形状已记录。本次仅诊断tokenize/模型/选择，未编码查询、未测ANN/权限复查或业务p95，不能据此宣称发布合格。

精度分配试验：data/hoj-legacy-precision-20261025保留旧selector实际需要的secondary mask为FP32，仅新增primary/fixed比较用BF16，复用request-local适配器和推理gate，没有题目/资料ID或经验分差阈值规则。6项主机检查通过，覆盖非平局排序反转、互斥分配/遗漏拒绝、超长与旧pool回退、fixed来源及同文档不同版本；在全部300个真实pool上只读调用生产selector的spy核对mask（无模型调用），274个评分场景的legacy mask逐项一致，其余26个短结果/单源/空权限跳过。该300 mask核对不是新的300问精度或耗时回归。

精度分配的真实模型诊断为17问/两重复，7个混来源问题每模式14次直接计时；split双重复的完整返回均与同轮全FP32一致，19/20必要证据进前三，旧v7-h008恢复[1,3]；全BF16仍[1,5]。同轮FP32/BF16/split直接段p95为558.66/330.54/385.34ms，包含两组都非空时重复tokenize全部pool的成本与真实forward形状；不能跨独立计时报告宣称稳定提速，更未测BGE/ANN/权限复查。最新诊断审计入口data/hoj-legacy-precision-20261025/final-state.json：anyCandidateQualified=false、splitFull300QualityValidated=false、splitBusinessLatencyValidated=false、candidateDeployed=false。旧selector中间比较保持FP32不等于最终排序保证，因为consensus仍比较两种精度的附加片段。下一步先验证这组精度分配的实际查询成本与完整非退步，达到原门槛后再处理生产生命周期；不将诊断或纯mask核对冒充已发布修复。

生产边界：所有实验均不接业务卷；最新审计仍为51运行源码匹配、三镜像保持、七服务/六healthy、原业务计数与零向量，实验容器按--rm清理，无SQL迁移/删卷/切分支/Git提交。当前双表示原型只实现retrieve，不能直接接生产：正式接线还需双投影幂等回填、撤销优先于物理清理、同一绝对deadline及最终入库receipt重读raw，避免外层持非重入ingest_lock；跨视图向量复用必须比较完整embedding输入。只有实际质量/耗时及生命周期验证完成才沿原授权部署；当前没有待用户确认的部署步骤。本人模型/真实源站AC-WA仍按用户决定延后，真实生成支持/拒答、未见课堂质量与完整七项目标仍未完成。

双表示与有限比较验证（2026-10-25，未部署）：继续按用户已确认的速度优先方向工作，原90%质量、逐题非退步及max(dense×1.5,dense+250ms)耗时门槛不变。原始与小节上下文两份向量表示共用一次真实query编码，候选仍使用原文/身份及原始向量cosine；上下文独有片段补取原始向量，payload不一致拒绝返回，分类器期间允许DELETE提交，最后同时重读两份manifest。只改data中的隔离原型和本页记录，生产src、镜像和业务数据保持。

完整实测：data/hoj-dual-view-20261025的19项主机合约检查通过，覆盖单次编码、并发不同查询、完整授权范围、原始cosine补取、混来源两次score回退及分类器期间撤销。有限缓存5问完整、1问缺真实raw分数而排除，c002解释/代码从[2,None]→[2,3]；该回放不作全量或性能证据。随后用原固定模型在断网临时容器重新编码文章、完整16问/四模式64响应，FP32/BF16均17/18前三、17/18前五、完整11/12；calibration7/8、heldout10/10，保住原前三证据和全部split聚合指标。原dense/current完整对象复现历史；共享一次查询编码和完整raw/context20候选均有记录。

耗时未通过：同轮dense/current/dual FP32/BF16 p95为273.50/482.19/905.86/561.07ms，原预算523.50ms，两个候选均拒绝采用。独立代理复核316条返回引用、1216条pool引用、32个mask、指标和哈希，与主代理一致。权威入口data/hoj-dual-view-20261025/final-state.json：validated仅为核对完成，publicCandidatePassed=false、fullRegressionQualified=false、candidateDeployed=false。首版审计误把topK上限当成必须恰好5条，scope只有4片段时失败；原脚本/日志保留，修正审计契约后补验，未重跑模型或改变检索结果。51运行源码、三镜像、七服务/六healthy、业务计数和零向量保持，实验容器已清理。

成本诊断与接续：data/hoj-dual-limited-20261025按代码名补召回后的完整来源tuple判断单源，单源沿用旧selector不调用分类器、vector补充独立，混源保留joint与超长回退；4项新增控制流检查通过。16问真实缓存×双重复中10题跳过、6题评分，BF16 batch4/2/1的直接选择段p95为268.10/268.89/267.04ms，完整输出一致但logit最大差0.03125；本次未见明确p95收益，保留batch4。计时含tokenize/CE/选择，不含BGE、ANN、向量补取和manifest检查，不能据减去CE时间宣称实际p95达标。单源策略较上个原型改变6题尾部ID，已暴露16问的逐组证据仍17/18，不称语义等价。

单源策略有效实测：hoj-dual-limited-keys-20261025实际完整16问/四模式64响应，17/18前三、17/18前五、完整11/12，原逐组前三及三split聚合非退步；10题单源零分类器调用，共享一次query编码、原dense/current历史对象均核对。BF16 p95 450.21ms，同轮dense238.61/current462.08ms、预算488.61ms，原全部门槛通过；FP32 614.86ms仍超时。这里只证明已暴露16问候选通过，不作全部场景/发布验收。权威入口hoj-dual-limited-keys-20261025/final-state.json：publicCandidatePassed=true、fullRegressionQualified=false、candidateDeployed=false。下一步已启动八组300问三模式完整回归，资料/非空允许范围查询重新编码；生产保持原版本。

实验返工保留：首次probe的dict keyword仍用dual_*、后续用limited_*，首题记录KeyError；原源码/报告/日志保留于hoj-dual-limited-20261025，这不是检索质量结果。只修实验模式名称后在hoj-dual-limited-keys-20261025另轮运行，模式键AST预检已通过；预检脚本首次错误地数为3处startswith，实际2处，失败发生在生成probe前，源码另存后修正。没有为无业务改动重跑Java/前端/544项主机测试。原旧284问完整回归尚未完成、本人模型/真实源站AC-WA继续按用户决定延后，真实生成支持/拒答、未见课堂质量及完整七项目标仍未完成。

此前候选与计算成本验证（2026-10-25，未部署）：用户已明确选择保持现有约0.5秒速度、继续提高证据完整性。原质量/逐题非退步及max(dense×1.5,dense+250ms)耗时门槛保持；不把优先级答复当成某个试验方案的部署批准。当前仍为a7a76b59…、15/18必要证据进前三，完整七项目标持续进行。最终证据入口为data/hoj-cpu-batch-20261025/final-state.json；validated仅表示主代理核对完成，anyCandidateQualified=false、goalComplete=false。

尾部排序回放失败：只固定前两条、对原3–5位按真实CE排序，精确保留原top5集合。旧284+诊断6问的290问实测logit回放虽将comparison-h1 [4,1]→[3,1]、rollback-snapshot-heldout 4→3、v8-h002 5→3，却使rollback-empty-union 3→5、v6-c007 3→4、v6-h008 [3,None,1]→[4,None,1]，拒绝采用，不通过逐题阈值拟合补洞。data/hoj-tail-order-20261025保留源码/计划/全部输出；该结果不是新编码、实际业务耗时或完整16问验证。

比较基准修正回放：先保留每源原始dense槽位，再补一位，primary额外位优先未覆盖heading；secondary仍用原cosine预算。实际current第三条必须在真实得分集合内才允许比较，唯一来源候选不再假定无须打分；之后沿用既有共识/多来源保护。data/hoj-incumbent-budget-20261025的290问缓存无原前三损失、原各split聚合非退步，comparison-h1与v8-h002改善，但v8-c002/c003仍缺，未进入src/镜像。回放的current是旧CE已完成结果，在线若实现须从共同预算的真实得分重放旧选择器，不能借缓存宣称已经只有一次物理forward或已达到耗时预算；本轮没有该在线实现或实际耗时结论。

批容量实测：同一固定FP32模型、CPU4/内存8GiB断网临时容器、六个缓存完整pool，每模式两次轮换。完整20对tokenize/18对active、原掩码/长度排序/动态padding/SDPA/inference_mode均保持，只改变物理batch常量。FP32 batch4直接分类器p95 1634.96ms，BF16 batch4/8/16为959.78/956.25/1122.05ms；4→8没有明确收益，16更慢。六题两次的完整返回字段均与FP32一致，logit并非数值等价。这仍不含BGE/检索/选择/最终权限复查，不是业务或发布耗时，也没有以调batch解决完整候选成本。

英文小模型诊断失败：按[官方模型卡](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2)选择英文、4.39M参数的TinyBERT作对照；官方metadata固定81d1926…，七个配置/分词器/safetensors文件约18.5MB按Git blob/LFS SHA校验，仅保存在独立模型目录，无pickle/远程代码/推理网络或中文效果声明。六个完整缓存pool各两次、同原上下文上限及18对掩码，沿用原共识选择后，三处漏项均未改善；直接分类器p95 179.09ms，首个形状约294ms已计入，没有剔除冷开销。已知六题仍缺三组，即使其余十题全部命中，整组最多15/18，不能达到原90%门槛，拒绝采用，不再补跑昂贵全量。首轮安装已完成文件/receipt却因末尾日志JSON参数错误退出，原日志/源码保留后离线复核、没有重复下载；首次模型已跑完但loadingInfo含set导致报告序列化失败，保存原快照，只修记录格式后在相同隔离环境补齐证据。最终report和原始失败均在data/hoj-tinybert-diagnostic-20261025，不能把首轮缺失报告伪补为完整成功。

边界与下一步：原公开文章/冻结证据/门槛不变；上下文组合缓存只够三处漏项的受限top5分析，其他问题有新增片段缺真实raw cosine或CE，不能补零或伪称完整组合。后续验证共享query向量的原始/上下文双表示与有限候选比较，必须重新获得完整候选和真实成本，保原前两条及问题级必要证据。辅助只读代理本轮因平台使用额度中止，新的回放/批容量/英文诊断最终由主代理核验，独立复核明确未完成；这是代理平台额度，非本地RAG模型额度。运行源码51文件、后端/RAG/Judge镜像、七服务/六healthy、业务计数和零业务向量均保持，实验容器已清理；无SQL迁移/删卷/切分支/Git提交。本人模型及真实源站AC-WA继续依用户决定延后，真实生成支持/拒答与未见课堂质量仍未完成。

最新结构切片验证（2026-10-25，未部署）：用隔离原型保护不超过1000字符的Markdown代码/展示公式原始区间，overlap不从保护块内部开始，保留连续原文覆盖、1000字符上限、最多150字符overlap；超长结构仍按原上限分割。最终原型在结构末尾结束当前片段并吸收可容纳的真实空白，让后续说明重新按段落组织。没有修改生产src、业务索引、镜像、冻结问题或证据分组，前端/后端接口与外发策略保持；完整七项目标继续进行。权威入口为data/hoj-markdown-block-edges-20261025/final-state.json，releaseQualified=false、goalComplete=false。

首版拒绝：data/hoj-markdown-blocks-20261025虽通过15项结构测试，但300问纯文本预检发现v8-c002解释组被拆散，不再能在单片段中完整命中；这应计为质量失败，没有改标签或运行昂贵模型。恰好1000字符测试首次漏算closing行LF（实际结构1001），失败源码/XML/log保留；修正为实际含LF的1000字符区间后通过。第二版曾因关闭块后的空行而回退，first-attempt保留，修正真实空白边界后不再出现该轮可避免回退。

最终预检与实测：16项主机结构测试实际通过、失败/错误/跳过0；八组300问纯文本预检全部原始证据组可达且回退0，旧七组284问的所有向量输入逐字不变。它不是300问模型回归。仅对原真实文章16问在现有FP32镜像的断网临时容器内重新编码原/新资料，四模式实际64份响应；原dense/current完整输出与发布基线一致，候选原文/来源定位/身份哈希、空允许范围均核对。Manacher 17→19片段，分块文章仍22片段；结构变化不是检索质量改善的充分条件。

候选仍只有15/18必要证据进前三、16/18进前五、完整9/12，与当前版本相同；首条8/18→9/18（解释2→1），但初始化代码/同块区间条件/哨兵解释三个漏项没有解决。all/calibration/heldout的原聚合指标均未退步，无已有前三必要证据损失；p95原dense/current/候选dense/候选CE为286.13/440.76/224.46/422.62ms，原预算536.13ms，耗时通过但原90%质量门槛失败，拒绝部署。四个无充分依据问题仍全部返回候选，不作生成拒答证明。不能为本轮非退步结果部署一个未解决质量缺口的索引重建，也不把旧284输入一致说成新模型回归通过。

结论与运行：边界完整性问题与互补证据排序问题需要分别处理；本轮未证明切片原型能修复目标漏项，下一步回到候选选择/互补证据排序，而非继续以结构调整替代检索验收。独立核对及最终审计重新计算原始指标/门槛，51运行源码和三服务镜像保持，七服务运行/六healthy、前后端API就绪，原业务计数与零向量保持，临时容器已移除；没有SQL迁移、删卷、切分支或Git提交。未为实验/文档变动重跑Java、前端构建或全量544项主机测试。本人模型与真实源站AC-WA依用户决定继续延后，真实生成支持/拒答和未见课堂质量仍未完成。

此前证据选择试验（2026-10-25，未部署）：在线仍为a7a76b59…，原始15/18必要证据进前三，质量目标未完成。该轮没有修改生产src、镜像、向量输入或冻结题集；只在断网临时容器和主机证据目录试验。最终运行/原始失败/实验缓存回归绑定于data/hoj-consensus-20261025/final-state.json，validated仅表示核对完成，anyCandidateQualified与goalComplete均为false。

已证实的改进与代价：受限全量CE共识保留当前首两条；其余授权dense20中最高两片段同来源且该来源已在当前top5时，只把最佳片段提升第3，保原cosine/原文和代码名覆盖。真实完整16问从15/18→17/18（94.44%），完整成功9/12→11/12，无原前三证据丢失，all/calibration/heldout的recall@1/3/5、MRR与成功率不退步；但同轮dense/current/候选p95为347.24/473.96/2780.45ms，候选超过原max(dense×1.5,dense+250ms)预算，拒绝部署。之前旧284+诊断6问的实测logit回放也未退步，并将comparison-h1 [4,1]→[3,1]；这是290问缓存、不是新的300问编码/耗时或未见泛化证明。

限量变体失败：对主资料按原题头未覆盖的小节分组，每组最多取两段，保留每来源m+1预算和原次要资料候选。首次回放虽然找回两个新证据，却使scc-c2、compare-mex-heldout、gap-comparison、v6-h008四个旧问题第3证据→4，拒绝采用。随后只禁止“题头已经多来源时再引入新来源”，允许集中题头扩展来源；290问回放无旧前三丢失，但完整16问实测暴露v8-h003 [1,2,3]→[1,2,4]，仅16/18（88.89%）、完整10/12，仍失败。同轮dense/current/候选p95为215.39/492.60/1003.84ms，实际最多forward五对仍超预算；不能以pair数代替耗时，或者以整体recall提高覆盖逐题必要证据损失。该轮current本身也比相对预算高约27.22ms，保留失败；历史单次性能通过不是持续负载保证，不据此重跑直到通过。

模型得分复用缺陷：独立只读核对发现实验CachedScores按文本字符串建字典；不同文档/版本完全同文时，inactive零可能覆盖真实负分。已用实际选择器最小例子复现：原应第4位b2却返回b1。另存data/hoj-score-identity-20261025，按完整文档/版本/哈希/chunkId身份和预期位置复用，校验真实forward状态；8项主机测试全部通过，含旧快照失败与新实现正确、跨版本同文、错位/未计算身份拒绝。旧实验快照/报告保持原样；修复仅在实验副本，没有据此将此前真实模型结果改标为“新代码已通过”，固定缓存后的候选还未重新资格验证，更未进入生产。

CPU低精度诊断（同日，未部署）：生产已启用按真实token长度排序、动态padding、batch=4和inference_mode；不把这些现有优化重复算作新实现。实际AMD Ryzen 7 8745H支持avx512_bf16，现有PyTorch 2.13.0+cpu可运行CPU BF16 autocast。用相同固定MiniLM和六个已缓存完整语义pool，轮换执行顺序、每模式各重复两次；每题20对完整tokenize、18对实际forward，使用完整身份对齐的实验得分缓存。直接分类器p95为FP32 1598.87ms、BF16 986.26ms（约减少38.32%）、现有旧接口dynamic INT8 836.24ms。BF16两次的全部返回字段与FP32相同，FP32亦与既有full-consensus完全一致；但最大logit差0.052535，不能声称数值等价或对其他题无损。INT8两次均丢失v8-c003和v8-h002的必要证据，拒绝采用。INT8接口已有弃用警告，仅作已安装接口的诊断，未增加依赖或转成生产方案。

该诊断不含BGE、检索、选择器和最终权限复查，不是完整16/300问质量或发布耗时测量；没有新编码、业务存储访问或模型外发。首轮脚本漏传active，实际forward20对但标记18，且质量汇总只取首次；原脚本/报告保留在data/hoj-cpu-rerank-20261025，不挪用该轮耗时。修正另存data/hoj-cpu-rerank-mask-20261025，实际传入active并逐重复汇总，独立核对通过；final-state.json重新核对报告摘要/原始输出/冻结资料和运行状态，anyCandidateQualified=false、goalComplete=false。BF16仍只能作为后续候选，不能据局部提速宣称完整业务达到现有预算。51运行源码、三服务镜像、七服务/六healthy、业务计数与零向量均保持，临时容器已移除；本轮仅实验和文档，不重复Java/前端构建或上一轮544项主机测试。

切片边界诊断（纯文本、未编码）：chunk-boundaries.json用当前生产分块方法重建全部43块，其中42块与已有诊断缓存逐字一致（另一块原缓存未覆盖）。确认Manacher公式和1166字符Mo代码块会跨字符边界，overlap亦可能从结构内部开始；未发现丢字，这不违反现有字符上限约定。v8-c002的483字符初始化代码块完整位于chunk13，整个634字符chunk还含150字符前文；解释证据位于chunk8，因此不能把该漏项归因于代码被截断。首个诊断断言错误地只接受```cpp并把代码块长度当作含overlap的chunk长度，失败脚本/日志已另存first-attempt；修正按实际fence区间核对后通过。下一最小原型可优先保护1000字符内的Markdown代码/公式原始区间并约束overlap，超长结构继续明确遵守上限；这只是结构完整性改善方向，不能预设解决排序。若采用，chunkHash/ID/序号将变化，需版本化并重建相关向量/FTS，保持原文连续覆盖和旧引用兼容，再跑冻结回归后决定，当前未修改生产分块。

互补证据边界：来源相同不代表同一问题需要的解释/代码都齐全；保留来源集合也无法保护同文档第三份证据。此前双表示缓存中v8-c002原排序[2,None]、上下文排序[4,2]有互补潜力，但其余两处仍未解决，缓存缺完整上下文top20与部分原始cosine，不能冒充完整双索引验证。下一步以问题级证据完整性为验收，优先验证不挤掉既有证据的上下文组合/覆盖约束及实际成本，保留原门槛。用户后续已确认保持现有速度，最新决定及证据见本页顶部；本人模型和真实源站AC-WA继续按用户决定延后，完整七项目标持续进行。

运行审计：51个运行源码与本地src一致；后端1c5b6027…、RAGa7a76b59…、JudgeServer ac1ae3d7…与前检查点相同，七服务运行/六healthy、前后端API就绪，业务计数3/8/9/6/0/1/0/0及零业务向量保持；两个实验容器已移除，无SQL迁移/删卷/切分支/Git提交。没有为本轮仅实验/文档变动重跑无关Java/前端构建或将上一轮544测试累加为本轮覆盖。

最新性能更新（2026-10-24，质量缺口仍保留）：本地RAG已由82aed835…更新为a7a76b59eb88…；后端1c5b6027…、JudgeServer保持。正常Dockerfile构建、544项实际主机测试（失败/错误/跳过0）、八组300问/900份实际响应，以及工厂/TCP/业务/Vue链路均已验证；本轮是性能和非退步改善，整体retrievalQualityPassed=false、goalComplete=false，待用户验收。权威入口为data/hoj-ce-budget-release-20261024/final-state.json。

诊断与实现：data/hoj-rag-profile-v8-20261024对六问记录实际BGE、tokenize、CE forward；三条慢查询有9/14/17个forward pair，CE约694–1248ms，证明这些查询主要耗时在CE计算。诊断不是发布耗时测量，也不推断所有负载均如此。通用预算对同一来源m个可替换原槽位保留已有候选，再按原cosine补至min(n,m+1)，topK=5最多实际forward六对。完整原pool仍全部tokenize，排除/固定项超512仍保留整次原结果；排序只纳入预算内logit，未forward的零占位不能胜过真实负分。首两条、主资料槽位、来源/版本/哈希顺序、代码名覆盖、原公开cosine/引用与最终撤销新读均保持；没有查询ID/资料名称特例，没有改变模型、切片、嵌入输入或索引代次。

正式镜像重新编码所有八组冻结资料/查询，比较dense、保存的原CE源码和新正式源码。前七组284问的原dense/CE返回ID、证据排名和cosine逐题与历史发布一致；八组在all/calibration/heldout的recall@1/3/5、MRR@5、完整成功@3均不低于原CE，无任何原前三证据丢失，原dense非退步也通过。新公开文章16问的正式port结果与先前隔离预算原型逐题一致。旧七组原绝对质量门槛通过；公开文章仍15/18（83.33%）前三、16/18前五、heldout9/10，原90%门槛明确未通过。所有八组耗时门槛通过；这不等于整体质量完成，也不改变既有失败标签/门槛。以下为本轮单容器CPU实测，非持续负载保证。

| 题集 / 问数 | dense p95 | 原CE p95 | 新CE p95 | 原预算 |
| --- | ---: | ---: | ---: | ---: |
| hoj-xcpc-v1 / 78 | 196.19ms | 399.58ms | 356.05ms | 446.19ms |
| hoj-xcpc-v2 / 66 | 188.37ms | 354.05ms | 357.16ms | 438.37ms |
| hoj-multiformat-v3 / 44 | 209.29ms | 389.90ms | 354.54ms | 459.29ms |
| hoj-scenarios-v4 / 24 | 268.36ms | 409.34ms | 413.26ms | 518.36ms |
| hoj-query-v5 / 24 | 269.30ms | 460.26ms | 428.84ms | 519.30ms |
| hoj-query-v6 / 24 | 326.47ms | 482.05ms | 493.30ms | 576.47ms |
| hoj-query-v7 / 24 | 355.66ms | 532.07ms | 496.82ms | 605.66ms |
| hoj-public-v8 / 16 | 258.15ms | 1294.39ms | 437.18ms | 508.15ms |

新资料同轮p95由1294.39ms降至437.18ms（dense 258.15ms，原预算508.15ms）。此前独立16问原型约1239.85→430.42ms只是阶段证据，正式发布以本轮完整测量为准。旧logit重放284问复现原选择器、仅5问无关片段ID变化且证据排名保持；1012→962只是forward对数，不作耗时替代。原主机测试曾因Windows沙箱默认pytest临时路径不可写而失败，失败日志/XML保留；显式独立basetemp后26项定向和完整544项通过，没有改生产或测试断言放宽错误。

实际链路：独立断网新镜像由真实from_env加载固定BGE/CE，通过loopback TCP签名请求；v6-h007完整证据[1,3]，空允许范围不调用CE，分类器运行期间DELETE先完成且迟到查询409。仅用齐base/dev/rag/video/rag-rerank五份overlay更新hoj-rag，身份初始化容器和所有挂载保持，51个运行源码逐文件匹配；七服务运行/六healthy，后端实际API和前端就绪。真实HOJ worker入库、显式发布、ADMIN隔离、旧版本/撤销、正式参赛403与Vue跟随引用均通过；临时12份讲义的sam-h2完整证据仍第3，四张真实页面截图已检查。原业务计数3/8/9/6/0/1/0/0和零向量恢复，临时账号/资料清理、撤销墓碑保留，无SQL迁移/删卷/Git提交。未重跑教材/ASR、虚拟赛或真实生成；本人模型与源站AC-WA仍按用户决定延后。

部署核对返工：首次脚本错误地直接比较Docker挂载数组顺序，服务已更新且健康，但验证在该处失败；原脚本与deployment.log保留。verify-deployment.py仅只读补验，同一容器连续六读实际观察到三种顺序，按目标目录比较六个卷/绑定源/读写权限均与五份Compose合并配置一致；身份初始化容器和卷的创建时间早于前检查点，源码/计数核对通过，没有再次部署。最终以deployment.json与verify-deployment.log为准。

下一步：处理已定位的证据选择缺口：v8-c002所需初始化代码在dense第8且CE不优先，v8-c003的正确条件在dense第11、CE可识别却受主资料槽位保护，v8-h002正确解释在dense第5、CE第一但排序固定。应验证通用互补证据选择，保留旧六问与新三处失败、冻结输入和原门槛；不能为通过当前题集写问题/来源特例。实际生成逐结论支持/拒答与未见课堂资料质量仍未完成，完整七项目标持续进行。

此前运行对照（2026-10-24，历史）：HOJ 后端完成 CF 数字题序及 AtCoder/牛客推荐、外部能力分类补全，随后修复训练管理伪造作者越权，现为 1c5b6027…；RAG 82aed835… 和 JudgeServer 保持。最新服务/数据/权限及训练Vue回归审计在 data/hoj-training-authorization-20261024/final-state.json；外部分类专项见 data/hoj-evidence-classification-20261024/final-state.json。下述 RAG 质量结果仍为 10 月 23 日专项证据，本轮没有重跑或扩大其质量结论。

此前质量检查（2026-10-24，历史实验已完成，候选未采用）：新增 [真实公开文章题集](../evaluation/hoj-public-v8/README.md)，三篇cp-algorithms原文固定至e12fc698…，保留原件/许可/来源摘要；独立source-only作者先写16问，逐组校验18组/41条原文证据并在首测前冻结。首测后该题集即为已暴露回归资料，不宣称后续调参仍属未见验证。当前dense与在线同版CE均只有15/18必要证据进前三（83.33%，原90%门槛未通过），heldout为9/10，前三完整成功为9/12可答题；四个无充分依据问题仍都有候选，不等同于生成拒答验证。三条漏项为Manacher边界解释与实际初始化代码互补、分块同块区间条件、代码哨兵/返回数组解释；原文/引用/权限范围校验均通过。

新资料耗时：首次独立容器中dense p95约278ms、当前CE约1475ms，超过原max(dense×1.5,dense+250ms)预算；后续同条件对照仍约260/1319ms。均为16问本地CPU测量，不作持续负载保证，也没有把差值直接认定为模型某阶段的耗时根因。当前选择器在这批资料没有改善必要证据排名。原七组通过仍是旧合成资料的历史事实，不能用它覆盖这次真实长文章的失败。

候选与失败保留：断网临时容器完整重测旧七组284问、逐题确认当前结果与原发布结果相同。放开全局排序/固定首一条/固定首两条三种候选分别使7/5/4个问题丢失原前三证据，均不采用。另一条预先定义的有限3/4同源换位在旧logit重放只改善rollback-snapshot-heldout的4→3，无旧前三丢失；重放不是新耗时测量，实际新资料首测也没有改善，未进入生产。首轮全量脚本在空授权范围用例因实验缓存未清空而中止，失败保留；修正后的新目录完整重跑，实际空范围与所有实验候选均为空，未把脚本错误归于生产授权。

同标题后续片段补首段上下文也仅在隔离模块试验：旧七组所有模型输入逐字相同，新题集21个向量输入改变，原文/切片/位置不变；真实重新编码后，初始化代码从前五缺失进入第2，但其互补边界证据从2降到4，前三仍15/18、前五16/18→17/18、首条8/18→7/18，且耗时失败，不能采用。生产src、镜像及索引代次均未修改；后端1c5b6027…、RAG82aed835…、JudgeServer保持，51个运行源码匹配，七服务/六healthy、原业务计数及零业务向量保持，所有临时容器移除。权威证据为data/hoj-public-baseline-20261024/final-state.json、report.json，data/hoj-ce-anchor-retry-20261024/report.json与continuity-report.json，以及data/hoj-section-context-20261024/report.json/projection-scope.json。final-state的validated仅表示证据/运行核对完成，retrievalQualityPassed与goalComplete明确为false。

下一步先记录三个新漏项的授权dense前20候选、真实CE分数与分阶段耗时，区分未召回/错误选择/互补证据被挤占；再验证通用的候选预算与多片段选择。保留所有冻结输入、原门槛和原失败，不增加问题ID/来源名称特例；真实生成逐结论支持/拒答、课堂资料质量、本人模型和源站AC-WA按既有边界保留，完整七项目标继续进行。

上一检查点（2026-10-23，已本地更新，待用户验收）：本地RAG已从767e8d2b…更新为82aed8351e9c…，后端a7660e51…及JudgeServer不变。正常Dockerfile构建，51个运行源码逐文件匹配；完整主机536项通过（失败/错误/跳过0），七组284问的28项原质量/非退步/耗时门槛全部通过，568份dense/CE响应与原型的证据排名、原文/位置/身份字段及cosine完全相同。最终运行审计7服务运行/6healthy，后端用真实API验证，原业务计数3/8/9/6/0/1/0/0和零向量恢复，测试账号/资料清理、撤销墓碑保留；未迁移SQL或提交Git。权威原始证据在data/hoj-ce-release-optimized-20261023/final-state.json。

实现：固定mMARCO MiniLM只在授权dense shortlist内选择同一份次要资料的片段，保留前两条、主资料全部槽位和完整文档/版本/哈希槽位顺序；不加入查询特例，不改变标签、topK、90%门槛、原文或公开cosine。all recall@3为228/237→231/237，heldout为110/119→113/119；v2 heldout为23/26→24/26，v6/v7为9/11→10/11。未配置模型时继续原dense/hybrid；显式配置必须验证固定七文件尺寸/SHA256、标准架构、本地safetensors及启动预热，不信任安装manifest自报，不下载远程代码。完整pair超过512 token整次保留原结果，模型错误/超时不静默回退。BGE与CE共享推理owner，模型工作在存储锁外，返回前新读READY/哈希防止迟到引用。

耗时返工保留：data/hoj-ce-release-20261023的首轮正式port虽然排名等价，但v4/v6/v7超过原耗时预算33.03/5.45/32.73ms，未部署。九问阶段诊断显示主要耗时为模型，存储等差额约9–13ms；第一条CE含冷启动，诊断不是发布门槛或原故障根因证明。现跳过“同来源只有一个原槽位、唯一候选正是原片段”的无效forward；混合pool仍完整tokenize所有pair，保留512整次回退语义。新代码经536项测试、完整284问和独立静态核对后通过原门槛，未覆盖旧失败报告。以下为新一次实际测量；最小预算余量16.30ms，不表示持续负载性能保证。

| 题集 / 问数 | dense p95 | CE p95 | 原预算 |
| --- | ---: | ---: | ---: |
| hoj-xcpc-v1 / 78 | 186.97ms | 420.67ms | 436.97ms |
| hoj-xcpc-v2 / 66 | 181.42ms | 358.88ms | 431.42ms |
| hoj-multiformat-v3 / 44 | 189.03ms | 373.12ms | 439.03ms |
| hoj-scenarios-v4 / 24 | 237.11ms | 407.84ms | 487.11ms |
| hoj-query-v5 / 24 | 236.71ms | 433.02ms | 486.71ms |
| hoj-query-v6 / 24 | 290.87ms | 501.20ms | 540.87ms |
| hoj-query-v7 / 24 | 304.73ms | 499.80ms | 554.73ms |

真实链路：独立断网Linux镜像通过实际from_env/离线预热/TCP签名HTTP，v6-h007调用真实CE后证据[1,3]；CE执行期间另一HTTP删除已完成，迟到查询409。部署后实际HOJ后台worker/BGE/Qdrant/显式发布和Vue页面验收通过，临时12份v2讲义经真实Java检索使sam-h2完整occ=0/从大到小证据排第3，并在真实页面检索、跟随原文引用；普通资料流程另验ADMIN隔离、坏新版本保留旧发布、修订发布撤销旧引用、可见性撤回、未配置本人模型拒绝及正式参赛403。首轮网站脚本将完整headingPath与末级小节名直接相等而失败，已按冻结原文完整路径修正断言，产品未改；live-smoke原失败/清理记录保留。最终四张实际页面截图归档于live-smoke-final及ce-live子目录，无API替代、无用户聊天模型调用。当前live没有重新跑虚拟赛/视频/全书ASR，沿用前轮专项证据。

教材范围：最早实验driver在七组完成后因Windows/Linux包版本身份断言中断，原report保持executionComplete=false/cleaned=false/book=null。最终正式代码单独在原缓存依赖环境回放367片段/四问[5,5]/[2]/[1]/[2]通过，引用/分数相同、CE调用0，无新全书或查询编码，也不证明当前Linux嵌入等价、全书CE质量或成本。请引用本轮book-replay-host.json；本轮七组主report的bookSupplement仍指向旧原型回放，主candidateGates只聚合七组，教材补证与部署/业务证据由final-state独立绑定，未改写原报告。

运行接续：可选compose.rag-rerank.yaml增加只读模型挂载与HOJ_RAG_RERANK_MODEL_PATH，本地.env.rag.local仅新增模型目录；后续更新需继续带齐compose.yaml/dev/rag/video/rag-rerank五份，保留数据/媒体/凭据卷。本轮模型来源为[固定官方revision](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/tree/1427fd652930e4ba29e8149678df786c240d8825)，七文件共492,748,252字节，经官方LFS/Git及代码固定SHA核验，未下载.bin/远程代码；安装证据在data/hoj-crossencoder-20261023，安装回执的deployed=false是下载时历史状态，实际运行以本轮deployment/final-state为准。

剩余质量条件：仍有六问未取得全部前三证据，无依据问题74/76仍有候选，尚未验证生成答案逐结论支持/拒答；logit和cosine均不作回答置信度。v1–v6已观察，v6作者曾收到一句算法摘要；v7为新source-only问题首测，但资料仍是合成旧来源，未见文档/真人课堂质量与大媒体成本没有完成。本人模型/真实源站AC-WA按用户选择延后，完整七项目标继续进行，用户验收未替代。

历史失败保留：第一条保护的同资料词法候选曾使v2到26/26，却损失v5队列第2条证据，拒绝；前两条保护的词法变体仍在v6低于90%；逐词MaxSim在v6仍9/11且v1/v2耗时超预算，也未采用。以下为这些隔离实验及此前阶段的详细记录，不能当作当前正式实现已失败或已部署的声明。

逐词与诊断证据：data/hoj-token-slots-20261023 的主机首轮在首个基线查询发生真实120秒deadline失败，未延长门槛；临时目录实际清理已由独立 runtime-audit 核对。分阶段复测常规查询约0.11秒，未复现先前超时，仅冷导入 / 预热约54.7秒，不能据此断定原故障根因。data/hoj-token-slots-linux-20261023 完整260问、真实本地模型 / 已安装投影头 / float16 SQLite校验、executionComplete/cleaned=true，仍拒绝采用；其SQLite位于宿主绑定证据目录，且包装器增加重复 READY 扫描 / top20 检索，不能把p95差异直接归因模型，连接/检索复用及本地临时盘优化尚未实测，也没有为性能已失败方案省略完整性校验。data/hoj-focus-diagnostic-20261023 的两问拆句定位完成，但不是启用门槛：回滚正确小节仍只排第二；mex定义可在单句进入第一，原文档槽位约束会限制加入。默认代码 / 业务库 / 向量未改，最新只读审计7服务运行/6 healthy、50个运行源码匹配、原题目/提交计数保留。

首个 shortlist 候选：保护首条及主资料全部槽位、仅在语义候选内调整次要资料，虽使 v2 23/26→26/26，原 v1/v3/v4 无前三丢失，但 v5-h008 第 2 条队列相等元素证据被删除，v5 heldout 11/11→10/11，拒绝部署。该候选 519 项主机测试通过只证明契约和回归行为，不能覆盖语义质量。独立镜像的 Linux helper 在持有共享锁时等待另一线程获取该锁，30 秒后超时，此为脚本设计错误，保留失败，不称 Linux 验收通过。该候选源码 / 新增测试已从默认路径移到证据目录；未部署的镜像标签保留，不复用为通过镜像。

阶段证据：`data/hoj-passage-choice-20261023` 保留未限制语义候选的 v3 MEX 退步；`data/hoj-semantic-passage-20261023` 为四集通过的 shortlist 隔离实验；`data/hoj-guarded-release-20261023` 保留正式实现的五集真实 CPU BGE 测量、v5 失败、519 项 XML、未部署镜像与 Linux helper 失败；`evaluation/hoj-query-v5/freeze.json` 冻结 24 个新问题及十份原合成资料。新问题不等于未见文档或真人课堂质量；原 v1–v5 已见结果后只作回归。未改原 90% 门槛、标签、分组、模型配置或 score 含义，没有调用用户模型 / 源站。

上一检查点（2026-10-23）：原生 PDF 按页校订已本地更新，后端 a7660e51…；原生 / OCR 真实管理页面校订、原件 / 旧发布保留、显式新发布、校订正文检索 / 引用与清理均通过。本轮实际后端 124 项测试、Java 8 / Node 22 构建及受控页面 26 组 / 19 图通过，178 页 / 221,894 码点编辑器输入与放弃草稿单独验证。RAG 767e8d2b… 及其 50 文件源码保持，前阶段完整 Python 507 项、整本实际导入 556 秒 / 367 片段 / 四问完整证据前五命中、原页定位及视频全流程的证据继续有效，本轮没有重复整本编码或视频 ASR。最终只读审计通过，原数据 / 卷保留，无新 SQL 或 Git 提交；后端用 running + 实际 API 验证就绪，未把不存在的 Docker Health 当作通过。校订入口不表示公式自动恢复；原 v2 recall@3=23/26 仍低于原 90% 门槛，真实模型逐结论支持 / 无依据拒答与未见课堂资料质量未验收。本人模型 / 真实源站 AC / WA 按用户决定延后，完整七项目标继续进行中，用户验收待进行。

下面保留 2026-10-19 上下文投影与混合排序阶段的证据。新题集只证明检索和来源定位的有限范围，不证明真实模型的逐结论支持、无依据拒答或真实课堂资料质量；完整 RAG 与七项目标仍未验收完成。

## 方法与固定资料

[原创 v1 题集](../evaluation/hoj-xcpc-v1/README.md) 含 12 份讲义、78 个问题。54 个有依据问题对应 56 组文档 / 实际小节 / 完整证据字符串，24 个无依据问题包含未覆盖的相近知识、域外内容和授权范围排除。calibration 与 heldout 各 39 题（27 有依据、12 无依据），首次测量前固定。题名命中、返回任何片段或只含一部分证据不计为命中；多证据问题另统计全部覆盖。

`scripts/evaluate_hoj_retrieval.py` 使用生产 KnowledgeIndex / 切片器、真实本地 CPU BGE-M3、独立临时磁盘 Qdrant；不访问现有 HOJ MySQL / 索引、不加载原型 `.env`、不调用用户模型。每条返回核对授权文档、版本 / 哈希、实际原文 / 小节、chunkHash 与 chunkId。报告保留每题排名、片段与分数，以及数据 / 模型配置哈希；运行器已补后续运行的实现文件哈希。纯指标单测与真实语义结果分开。

## 实测结果

两轮题集 / 标签 / 模型配置相同，临时索引均关闭清理。基线包含 48 个片段；上下文投影 v2 为 36 个正文片段。

| 指标 | 原正文投影 | 标题 / 小节上下文投影 v2 |
| --- | ---: | ---: |
| 全体 evidence recall@1 | 71.43% | 92.86% |
| 全体 evidence recall@3 | 96.43% | 98.21% |
| 全体 evidence recall@5 | 96.43% | 100.00% |
| 全体 success@3（全部证据覆盖） | 96.30% | 98.15% |
| 全体 MRR@5 | 0.8519 | 0.9784 |
| 原 heldout 组 evidence recall@3 | 92.86% | 96.43% |
| 无依据问题仍有候选 | 24 / 24 | 24 / 24 |

这里的百分比分母是标注证据 / 问题，不是生成答案正确率。基线 heldout 的缺失用于诊断：Fenwick 正文缺少文档主题上下文，纯标题片段占用跨文档比较题的前 5 位。修改后在同集回归中改善；因为这些结果已被观察，修正后的原 heldout 组不再是未见样本泛化证据。两次没有改题、标签、分组、删失败项或下调 0.90 门槛。比较题仍有一个来源排第 5，未宣称 top3 全覆盖。

两轮 calibration 都找不到同时满足证据保留 >= 0.90、无依据候选放行 <= 0.05 的单一分数阈值；有依据与无依据分数重叠。没有部署分数阈值，不能声称检索能自动判断“资料足够”。纯检索只给候选；用户明确允许模型后仍要求模型返回已有引用，空引用转为依据不足提示。真实模型的逐结论支持和拒答质量仍未评测。

## 实现与兼容

`KnowledgeIndex.chunks` 不再把仅由 Markdown 标题 / 空行组成的小节单独索引，正文、代码与实际标题路径仍保留。向量输入加入已验证资料题名与小节路径，返回的 excerpt / chunkHash 始终是原文，不把新前缀伪装成原文。

内部 payload 加 embeddingRevision=2；旧 READY 资料仍可按旧投影检索，不自动覆盖。管理员显式重建同版本时，先检查当前投影与整个版本的点位数量；旧投影或残留点必须重建，不能仅凭原 READY / 一致计数返回完成。正文 / 标题 / 格式的不可变约束不变。新向量计算后重查撤销状态，在非可检索的 WRITING 阶段清理该完整版本范围的旧点，再写入并发布 READY；其他资料 / 版本 / namespace 不受清理影响。更新镜像本身不自动重建既有资料，已有库可由管理员对相关版本点“重建索引”。无新 SQL 迁移。

## 验证与部署边界

- 最终全量 Python 实际执行 294 项通过，失败 / 错误 / 跳过均 0，已核对 XML 中实际 testcase 数；有 1 条既有 Starlette TestClient 弃用提示。包含先前 293 项上下文修正和随后补充的“当前投影数量正常但存在额外旧点”保护；早期纯本地指标 / 存储 9 项结果保留为阶段证据，不重复相加。
- 两轮真实 BGE-M3 语义测量均 executionComplete=true、cleaned=true，默认检索门槛通过；阈值可分性未通过，语义生成未运行。首次运行器误把 QdrantClient 当 context manager，错误报告已保留；改为 try/finally 显式关闭后才计真实测量通过。
- 最终镜像 sha256:30d8645ccd283420e351ef4c4547aad30c196f929ee4bcaec8880de3f3f84f33 已构建并用于 RAG / 初始化容器；部署后读取容器 index.py 哈希，与最终工作区一致。服务凭据身份保留，未打印凭据，索引卷保留。早期自动审批因额度限制失败时没有执行切换；审批恢复后才重新验证并更新。最终构建首次遇到 Docker Hub token EOF，保留失败日志后同命令重试成功，没有绕过审批或删除卷。
- 本轮新真实闭环 4 组通过：上传 Markdown / TXT、持久 worker、本地 BGE-M3 / Qdrant、显式发布、PUBLIC / ADMIN 隔离、真实解析拒绝与旧发布保留、修订版替换、旧引用失效、撤回和正式参赛限制。实际 Vue / API 无拦截的 2 图已查看；携带测试向量重启后 healthy、点位保留、已发布 v3 直接检索且不重新入库。validated / cleaned / vectorPointsRestored 均 true，未用 SQL 伪设 INDEXED，没有外站代码投递或真实用户模型调用。
- 最终 7 服务运行、6 健康探测 healthy，初始化容器退出 0；前后端 HTTP 200，RAG 无主机端口。原编程题 3 / 客观题 8 保留，知识库四表 / 临时用户 / 临时库及向量点位 0，manifest 的 18 条 DELETED 撤销墓碑按设计保留。后端 / JudgeServer 镜像与前端源码未改，两个仓库原分支与 HEAD 保留，未提交 Git。

本地证据归档在 `D:\learning-OS-RAG\data\hoj-evaluation-evidence-20261019`（已忽略，不提交生成报告）：两轮完整 JSON / 日志、早期失败报告、最终 294 项 XML / 日志、构建与部署、live/ 四组真实报告及 2 图、final-state.json 和源码哈希；早期 293 / 9 项证据独立保留。整体进度的权威入口仍为 HOJ `docs/docs/ai/TODO.md`。

## 下一步

部署与真实复验已完成；接下来建立新的未见问答集，评估混合检索或本地重排；这也是 [BGE-M3 官方模型卡](https://huggingface.co/BAAI/bge-m3/blob/main/README.md) 建议的后续检索路线，尚未作为本项目依赖引入。真实本人模型仍按用户选择延后，不重复索取 Key。PDF / OCR / 视频和完整七项目标其余事项继续保留。

以下保留本机复验命令，供后续源码变更时使用。本次最终 294 项已实际通过；未来运行仍需核对新的 XML，不能沿用本次计数证明新版本通过。

```powershell
Set-Location D:/learning-OS-RAG
$taskStamp=Get-Date -Format yyyyMMddHHmmss
& 'C:/Users/lenovo/AppData/Local/Programs/Python/Python312/python.exe' -m pytest tests `
  --basetemp="D:/learning-OS-RAG/data/pytest-hoj-final-$taskStamp" `
  --junitxml=D:/learning-OS-RAG/data/hoj-final-python.xml
```

全量通过后，在 HOJ Git 根目录执行：

```powershell
Set-Location D:/IdeaProjects/HOJ-master
docker compose --env-file HOJ-master/.env.rag.local -p hoj-master `
  -f HOJ-master/compose.yaml -f HOJ-master/compose.dev.yaml -f HOJ-master/compose.rag.yaml `
  build hoj-rag-identity
docker compose --env-file HOJ-master/.env.rag.local -p hoj-master `
  -f HOJ-master/compose.yaml -f HOJ-master/compose.dev.yaml -f HOJ-master/compose.rag.yaml `
  up -d hoj-rag
docker compose --env-file HOJ-master/.env.rag.local -p hoj-master `
  -f HOJ-master/compose.yaml -f HOJ-master/compose.dev.yaml -f HOJ-master/compose.rag.yaml `
  ps -a
```

等到 hoj-rag 显示 healthy，确认初始化容器正常退出 0、后端仍可用，再运行真实闭环。该脚本创建 / 清理随机测试账号与资料，并重启本地 RAG 验证向量保留，不调用聊天模型或源站提交；不能用旧的 INDEXED 夹具脚本代替。

```powershell
$env:HOJ_LOCAL_KNOWLEDGE_RAG_TEST='1'
$env:HOJ_KNOWLEDGE_RESTART_TEST='1'
$env:PAPER_REVIEW_OUTPUT='D:/learning-OS-RAG/data/hoj-evaluation-evidence-20261019/live'
& 'C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' `
  HOJ-master/hoj-vue/tests/knowledge-rag-local-integration.cjs
```

随后核对新报告 validated / cleaned / vectorPointsRestored / restart、实际镜像和健康状态。保留已有数据卷与凭据；已有非测试资料只在管理员显式重建时更新投影。

## 关键词 / 向量实验：未通过默认启用门槛

新增 evaluation/hoj-xcpc-v2：12 份原创合成讲义、66 个问题，calibration / heldout 各 33；50 个有依据问题 / 52 组完整证据，16 个无依据。材料由编码助手编写，未声称真人查询或专家独立复核。在实现混合排序、观察排名前校验并写 freeze.json；运行器每次验证全部冻结哈希。固定数据未因失败修改，原 v1 也保持不变。此次新 v2 的首次对比完成后已经观察 heldout，后续修改只能称同集回归，需另建未见样例验证泛化。

实现复用 [SQLite FTS5](https://www.sqlite.org/fts5.html) 的外部内容表 / BM25；中文按字 / 相邻双字构造词项，英文和下划线标识符保留，词项转固定 ASCII 编码并参数化查询，输入不能变成 SQL 或 FTS 运算符。查询最多 64 个唯一词项，候选最多 100；完整 namespace / documentId / version / contentHash 和 READY 状态在排序截断前筛选。关键词仅返回身份，原文及哈希仍取 Qdrant。以 [RRF](https://qdrant.tech/documentation/search/hybrid-queries/) 合并两路（本实现显式常数 60、等权），返回的 score 仍为 cosine，而非 RRF 分数或回答置信度。

关键词投影和 READY 回执同一 SQLite 事务发布；失败回滚为不可检索状态，撤销先原子删除关键词再清理向量，墓碑禁止迟到复活。旧 READY 缺关键词时仍可走 dense；显式重建可只补词项，不重算齐全向量，亦修复额外 / 不同哈希的旧词项。代码的 HOJ_RAG_RETRIEVAL_MODE 可取 dense / hybrid，默认 dense，浏览器不能选模式。本轮未改运行 Compose 或 HOJ 业务接口。

四次真实本地 BGE-M3 测量使用同一实现 / 模型配置与分别相同的两套数据，均 integrityPassed / executionComplete / cleaned=true，无现有 HOJ 数据或模型 API 访问。比较器拒绝数据、源码、模型、运行依赖或分块不同的报告。采用保守门槛：all / heldout 的 recall@1/3/5、success@3、MRR@5 均不退步；p95 不超 max(原 p95×1.5, 原 p95+250ms)。门槛没有因下列退步调低。

| 数据 / 指标 | Dense | 等权混合 |
| --- | ---: | ---: |
| 新 v2 recall@1 | 78.85% | 78.85% |
| 新 v2 recall@3 | 94.23% | 98.08% |
| 新 v2 recall@5 | 96.15% | 100% |
| 新 v2 heldout recall@3 | 88.46% | 96.15% |
| 新 v2 p95 | 329ms | 328ms |
| 原 v1 recall@1 | 92.86% | 89.29% |
| 原 v1 recall@3 | 98.21% | 100% |
| 原 v1 MRR@5 | 0.9784 | 0.9599 |
| 原 v1 p95 | 281ms | 406ms |

新 v2 通过门槛，原 v1 因首条及 MRR 退步未通过，整体不采用。前缀和自然语言问题被二分讲义挤到第二；整数溢出问题被复杂度讲义挤到第二；英文 KMP 重叠匹配被同篇其他小节挤到第三。没有仅报 top3 增益、删失败题或临时加锚点掩盖问题，也未在观察这批 heldout 后调权重再称未见验证。无依据问题仍有候选：v1 为 24/24，v2 为 15/16（唯一空授权范围返回空），两模式相同；不增加拒答阈值、不宣称生成质量通过。

验证为全量 Python 325 项和其后新增签名 HTTP 专项 1 项，均通过（各有既有 Starlette 弃用提示）；全量与追加不重复相加为再次完整执行。Linux 候选镜像实际验证 FTS5、独立关键词召回、cosine 和删除。候选镜像保留为 hoj-rag:hybrid-eval-20261019，默认 hoj-rag:local 已恢复为在线 PDF 镜像，未切换服务或修改 MySQL / 用户资料。首次构建 Docker Hub token EOF 后同命令重试成功，失败日志保留。

完整 JSON、门槛比较、首条失败、325 项 XML / 日志、追加 HTTP、Linux验证、构建和标签状态在 data/hoj-hybrid-evidence-20261019（已忽略）。下一步在固定资料上评估能兼顾首条与多证据的重排，另补新的独立问答；OCR / 视频及完整七项目标其余范围继续推进，真实本人模型仍延后。整体权威进度见 HOJ TODO。

## 多格式新题集 v3：2026-10-20 首次测量

[hoj-multiformat-v3](../evaluation/hoj-multiformat-v3/README.md) 包含 Markdown / TXT / PDF 页码正文 / 视频字幕各 2 份，共 8 份原创合成资料、44 个问题。calibration / heldout 各 22；36 个有依据问题对应 40 组证据，8 个无依据问题包含近似知识缺失、现实信息缺失和限定授权范围。编码助手同时编写资料及标注，没有真人查询采集或专家独立审核；首次测量前冻结全部材料和问题，只在这次测量中称为新样例。观察过结果后只能用于同集回归。

PDF 样例是规范提取正文，保留空白第 2 页，证据在真实编号的第 1 / 3 页；视频样例是规范校订字幕，包含 60 秒之后的独立片段。这次不验证原始 PDF / OCR / ASR 的识别准确率。评测复用实际生产分块和本地模型，标题、格式清单、原文及问题都纳入指纹；冻结覆盖必须完整，不能只保护问题文件。旧 v1 / v2 不改动，Windows 旧冻结路径也在 Linux 加载验证通过。v3 单独设置 eol=lf，避免后续 Windows / Linux checkout 改变冻结字节。

使用现有 RAG 镜像在 network=none、根文件系统只读的独立容器执行。只挂载源码 / 评测脚本 / 固定资料 / 本地模型（只读）和本轮报告目录；不挂载线上数据卷、服务身份或个人 .env，不调用用户问答模型。14 个临时片段索引在容器 / 临时目录内，结束关闭并清理。所有返回核验完整文档身份 / 版本 / 内容与片段哈希、实际原文；PDF 页码与视频时间段额外回查源正文。位置校验不是生成结论的语义支持校验。

| 指标 | 全部问题 | 首次 heldout |
| --- | ---: | ---: |
| 问题数（含无依据） | 44 | 22 |
| 有依据问题 / 证据组 | 36 / 40 | 18 / 20 |
| evidence recall@1 | 72.5% | 65% |
| evidence recall@3 | 97.5% | 95% |
| evidence recall@5 | 100% | 100% |
| success@3（完整证据覆盖） | 97.22% | 94.44% |
| MRR@5 | 0.8912 | 0.8380 |
| 无依据问题仍有候选 | 7 / 8 | 3 / 4 |

| 来源格式 | 证据组 | recall@1 | recall@3 | recall@5 |
| --- | ---: | ---: | ---: | ---: |
| Markdown | 12 | 50% | 100% | 100% |
| TXT | 10 | 100% | 100% | 100% |
| PDF 页码正文 | 8 | 75% | 100% | 100% |
| 视频字幕 | 10 | 70% | 90% | 100% |

沿用 recall@3 >= 0.90 的 all / heldout 门槛，本次通过；没有因结果修改题集、标签、门槛、排序权重或线上配置。具体失败是 rollback-snapshot-heldout：有关同集合 union 不产生修改、按栈长度回滚的完整字幕证据排第 4，而第一条来自同一视频的“不做路径压缩”片段。命中正确文档并不意味着已覆盖问题要求，因此它仍是 top3 失败。跨格式题的排名在完整列表里计算，不先过滤来源格式制造靠前名次。

无依据检索仍有 7/8 返回候选（唯一空授权范围返回空）；calibration 再次找不到同时满足证据保留 >= 0.90、无依据候选放行 <= 0.05 的单一相似度阈值。没有部署阈值或把 cosine 当作回答置信度。线上生成仍使用用户明确勾选和本人模型，引用 ID / 来源位置校验与语义支持之间的缺口没有被本次评测消除。

最终全量 Python 实际 XML 核对 407 个 testcase，失败 / 错误 / 跳过 0，有 1 条既有 Starlette TestClient 弃用提示；22 项聚焦结果是前期证据，不重复相加。首次主机运行因缺少锁定 pypdf 依赖在收集阶段停止，失败报告保留；随后仅在本轮隔离 Python 环境从现有镜像复用同版 pypdf 6.21.1，补齐后才完整执行，未改全局环境或跳过 PDF 测试。聚焦运行结束时另有 pyreadline 控制台析构提示，实际退出 0 / 用例通过，日志保留。

结果 executionComplete / cleaned / integrityPassed / sourceLocationsVerified / retrievalGatePassed 均 true，semanticAnswerValidation=NOT_RUN_NO_CHAT_MODEL。本轮不需要构建或切换服务：7 服务运行 / 6 healthy、初始化退出 0、前后端 HTTP 200，在线 14 个知识库模块与源码一致，既有题目、空知识库 / 视频 / 向量状态和两个仓库 HEAD 保留。详见 data/hoj-multiformat-evidence-20261020（已忽略）的 dense-first.json、407 项 XML / 日志、linux-datasets.log、validation-summary.json、source-hashes.json、final-state.json，早期依赖失败与原始报告独立保留；未提交 Git。

下一步评估本地重排能否同时兼顾首条和多证据覆盖，须保持旧 v1 / v2 / v3 回归并另建未观察的留出问题；真实生成、逐结论引用支持、拒答以及真实课堂资料仍单独待验证。本人模型和远程 AC / WA 继续按用户选择延后，不重复索要凭据。整体七项目标保持进行中。

## ColBERT 词级重排：2026-10-21 隔离实验

复用原 BGE-M3 主模型，补同一 revision 5617a9f61b028005a4858fdac845db406aefb181 的官方 colbert_linear.pt（2,100,674 字节；LFS SHA-256 19bfbae397c2b7524158c919d0e9b19393c5639d098f0a66932c91ed8f5f9abb）。只下载这一份公开投影参数，不替换本机 Embedding 目录或请求聊天模型。投影 / 去除 CLS 和 padding、保留 EOS、逐 token 归一化、对查询 token 的最佳匹配取平均，依据 [BGE-M3 官方模型卡](https://huggingface.co/BAAI/bge-m3) 和 [FlagEmbedding 推理代码](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/embedder/encoder_only/m3.py)。参数加载使用 weights_only=True 并先核对固定哈希。

scripts/hoj_colbert.py 和 evaluate_hoj_colbert.py 仅为离线实验，服务没有导入它们。先用实际 KnowledgeIndex 在完整已授权版本范围取 10 个 dense 候选，同一候选列表前 5 作为基线；以题名 / 实际小节 / 正文投影重新编码并重排前 5，保留原 cosine 和全部引用身份，单列 colbertScore。数值采用 CPU float32，不设混合权重、拒答阈值、逐题特例或跨查询 passage 缓存；实际额外编码耗时纳入 p95，不能用预先缓存后的内积耗时代表当前系统的改造成本。

测量使用现有镜像的无网络、只读根文件系统独立容器，4 CPU / 6 GiB 上限、临时 tmpfs 索引，结束关闭并清理；只读挂载源码 / 脚本 / 冻结资料 / 模型，单独证据目录可写。不挂载在线卷、服务身份、个人 .env 或用户资料，不构建 / 切换服务。每条候选仍核验完整文档 / 版本 / 内容与片段哈希、实际原文 / 小节、PDF 物理页 / 视频时间；不把正确来源当作完整证据覆盖。

首轮 v3 的 44 问 / 40 组证据成对测量已 executionComplete / integrityPassed / sourceLocationsVerified / cleaned=true：all recall@1 从 72.5% 到 80%，recall@3 保持 97.5%，recall@5 保持 100%；原 heldout 组 recall@1 从 65% 到 70%，recall@3 保持 95%。MRR@5 从 0.8912 到 0.9375。原失败 rollback-snapshot-heldout 的证据从第 4 到第 2，但 mex-count-heldout 从第 3 退到第 4，保留全部逐题结果；旧数据已被观察，均是回归证据。无依据仍 7/8 有候选，不能据此声称可自动拒答。

p95 从约 314 ms 到 6420 ms，超过沿用 max(旧 p95×1.5, 旧 p95+250ms)=564 ms；门槛不通过，**不采用**。新候选若未来要部署，仍须旧 v1/v2/v3 和新问题的质量 / 耗时共同通过。本次因采用门槛已失败，不再为部署补跑 v1/v2，也不冒称它们通过；可先研究入库时持久化词级向量的容量 / 权限 / 修订成本，或另选轻量重排器，但那是新的方案和测量，不能拿本轮结果直接部署。

[hoj-scenarios-v4](../evaluation/hoj-scenarios-v4/README.md) 在本次新问题测量前冻结：逐字节复用 v3 八份合成资料，新增 24 个中英数值 / 代码场景问题，20 个有依据 / 24 组证据，4 个无依据；两组各 12 问。编码助手编写，没有独立专家审核或新真实语料，只能称已知小型语料上的新问题。题名 / 格式清单、正文和全部问题的冻结覆盖已由生产分块 / 位置校验核对；没有改变旧集、标注或排序参数，观察本轮结果后不再称未见问题。

| 测量 / 指标 | Dense 基线 | ColBERT 重排 |
| --- | ---: | ---: |
| 旧 v3 all recall@1 | 72.5% | 80% |
| 旧 v3 heldout recall@1 | 65% | 70% |
| 旧 v3 all / heldout recall@3 | 97.5% / 95% | 97.5% / 95% |
| 旧 v3 all MRR@5 | 0.8912 | 0.9375 |
| 旧 v3 p95 | 314ms | 6420ms |
| 新 v4 all recall@1 | 75% | 79.17% |
| 新 v4 首次 heldout recall@1 | 66.67% | 75% |
| 新 v4 all / heldout recall@3 | 100% / 100% | 100% / 100% |
| 新 v4 all MRR@5 | 0.9417 | 0.9667 |
| 新 v4 p95 | 354ms | 6067ms |

v4 的 quality gates 通过，但 p95 超过 604ms 门槛，整体仍不采用。并查集无变化合并的场景证据从第 3 到第 1；窗口重复元素场景从第 2 退到第 3，保留退步。两轮 recall@5 均为 100%，这不是完整回答正确率；v4 四个无依据问题仍全部有候选，本实验没有提供无依据拒答能力。两个报告均 executionComplete / integrityPassed / sourceLocationsVerified / cleaned=true；门槛失败以进程退出 2 表示，failure=null，不混同于执行异常或通过。

最终单批 37 项实际 testcase 通过、失败 / 错误 / 跳过 0（数值 / 头文件哈希、CLS / EOS / padding、来源 / cosine 保留、稳定并列、耗时 / 首条退步门槛和冻结集验证）；阶段 36+1 不重复相加。主机结束时有既有 pyreadline 控制台析构提示，保留日志，实际退出 0、XML 已核对。候选脚本未被生产服务导入，无需为离线脚本重建镜像或扩大到后端测试。

本轮证据位于 data/hoj-colbert-evidence-20261021（已忽略）：head-provenance.json、python-final.xml / 日志、v3-first 与 v4-first 完整报告 / 日志、v4-container-scope.json、validation-summary.json、final-state.json、source-hashes.json 和已实际通过的 audit-final.py。v3 跑完后仅给运行器补充 dataset-usage 声明，以区分旧回归与新问题，数值计算 / 候选数 / 门槛文件未变；各报告记录实际源码哈希，不把两个脚本版本假称同一字节。线上保持 dense 和原镜像，14 个运行 RAG 模块与源码一致；7 服务运行 / 6 healthy、初始化退出 0，原题目、空知识库 / 视频、原 CF 失败导入和两个仓库 HEAD 保留，未提交 Git。

本节结束时的下一步是验证“入库时同时产生并持久化词级向量”的容量、修订 / 删除 / 撤权隔离及冷启动，后续实际结果见下节。本人模型 / 生成公式与语义质量、真实资料和完整七项目标仍未完成，进度统一见 HOJ TODO。

## 持久化词级索引：2026-10-21 候选实现，未采用

新增 late_interaction.py / token_index.py，KnowledgeIndex 增加显式内部 colbert 模式。BGE-M3 同一次 encoder forward 输出原 dense 和词级向量，dense 复用实际 Pooling / Normalize 模块，避免依赖 Sentence Transformers 5.6 已移除的旧池化属性。指纹绑定主权重、tokenizer、配置 / 池化、投影头和 codec 代次；投影头先核对固定 SHA-256，再从已验证字节 weights_only 加载。若未来改变编码 / 投影语义，须升级该代次并显式重建，不能继续复用旧向量。

词级向量以 little-endian float16 数值 BLOB 保存在同一 SQLite manifest 库，逐块记录形状 / SHA-256，键包含 namespace / document / version / contentHash / chunk / 模型指纹。没有 pickle 或第二份原文。token 与 READY / 关键词在同一事务发布；任意失败整体不变为 READY，删除先提交 DELETED 并清理 token，再清理 Qdrant。真实 SQLite 多连接的并发写入先取得 writer 锁再检查全局配额，不重复花费剩余空间。

默认 512 MiB 限制的是活跃 token payload 总量，跨 namespace 计算；SQLite 页、空闲页和事务日志另有物理开销，不声称整个目录最多 512 MiB。单块最多 8191×1024 个 fp16 数值，加载前限长、验形状 / 哈希 / 有限性。容量超限返回 TOKEN_INDEX_QUOTA，不发布不完整版本；删除释放逻辑额度，SQLite 可复用空闲页。当前仅测量合成语料，尚未验证最大 200 页 / 20 万字资料、长期碎片或生产容量策略。

查询只编码问题，在完整允许版本范围内取固定 10 个 dense 候选，按当前指纹读取 token，重排到请求的 topK。缺少旧 / 部分投影时整次保留 dense 顺序，不把不同模型分数混排；管理员显式重建后才使用当前投影。损坏数据报错，显式重建会逐块校验并修复同长度损坏；重排后再次读取版本状态，另一 adapter 的撤回不能产生迟到结果。公开返回的 score 仍是 cosine，不伪装成重排分数或回答置信度。

最终源码的四组实际本地 CPU 成对回归共 212 个不同问题。每组采用独立磁盘 Qdrant / SQLite，dense 与组合编码的 query / passage 实测最大绝对差均为 0，恢复后的查询只出现问题编码、无资料重编或 dense fallback。model / 脚本 / 索引实现哈希、逐题原文 / 版本 / 位置校验和存储开销均保留。以下全是已观察题集的回归，不重新称未见测试。

| 题集 | all recall@1：dense→token | all recall@3：dense→token | p95：dense→token | token payload / SQLite |
| --- | --- | --- | --- | --- |
| v1，78 问 | 92.86%→91.07% | 98.21%→98.21% | 276→282 ms | 7,567,360 / 7,786,496 B |
| v2，66 问 | 78.85%→78.85% | 94.23%→96.15% | 205→221 ms | 5,097,472 / 5,251,072 B |
| v3，44 问 | 72.5%→80% | 97.5%→97.5% | 234→229 ms | 3,520,512 / 3,665,920 B |
| v4，24 问 | 75%→79.17% | 100%→100% | 276→281 ms | 3,520,512 / 3,665,920 B |

四个耗时门槛均通过；v2/v3/v4 的完整门槛通过，v1 未通过。v1 heldout recall@1 从 92.86% 到 89.29%，all MRR 从 0.9784 到 0.9722、heldout MRR 从 0.9815 到 0.9630。具体首条退步包括静态区间和被同篇差分小节挤走、不可达距离被 BFS 总论挤走、逆元条件被快速幂小节挤走。没有调权重、门槛或删题来掩盖这些退步，**不采用 / 不部署**。无依据候选仍不能作为可回答证明，真实模型拒答和语义支持并未验收。

两个独立 Docker 容器分别构建 / 读取同一合成磁盘夹具，容器 ID 不同；读取进程重新加载并预热模型约 23.44 秒，首个查询约 158.69ms，无资料重编，允许版本 / 旧新版本隔离与撤回通过。预热先于查询计时，未驱逐操作系统页缓存，不把一次首查询当作 p95。原先按完整浮点字典精确比较失败，逐字段核对后确认 cosine 差约 3.58e-8；来源字段保持精确比较，cosine 按预先用于 dense 数值等价校验的 2e-6 容差验证。最终合成 token / 向量记录已删，带撤销墓碑的隔离数据库留作证据，不是在线资料。

最终全量 Python XML 实际 455 项通过、失败 / 错误 / 跳过 0，含现有服务 / PDF / 视频及新数值管线、真实 SQLite / 磁盘 Qdrant 生命周期、并发配额、namespace / 版本、撤回竞争与损坏修复。有 1 条既有 Starlette TestClient 弃用提示。默认沙盒全量曾挂在首个 HTTP 夹具，核对该进程和授权单项通过后，只终止那两个测试 PID，授权环境才计最终完成；未停线上服务或降低鉴权。首轮模型结构校验失败、同长度损坏重建的真实失败、454 项阶段结果及恢复比较失败均保留，不与最终计数相加。

证据在 data/hoj-token-index-20261021（已忽略）：python-full-final.xml / 日志、四份 v*-release.json / 日志 / scope、restart-build / restart-read、各容器 ID、validation-summary.json、final-state.json、source-hashes.json、已实际通过的 audit-final.py。早期测量与修复后测量分开，最终四组的源码哈希全部对应当前候选。

在线 RAG 仍为 6d4240ae… / dense，原后端和 JudgeServer 镜像保持；7 服务运行 / 6 healthy，初始化退出 0，前后端 HTTP 200。候选未部署：相对运行的 14 个模块，工作区 index.py 已改变，并新增两个模块，在线 SQLite 没有 token_chunk 表。service.from_env 仍只接受 dense / hybrid；HTTP / 浏览器不能选择新模式。本轮无 HOJ MySQL 迁移或主库写入，题目、旧 CF 失败导入和空知识库 / 视频保持，两个仓库 HEAD 未变、未提交 Git。

后续先回到七项需求核对用户可用功能与剩余缺口；这个性能问题已有可复用实现和量化证据，但重排质量未达到替换门槛，不能为继续优化而自动启用。若后续调整排序算法，仍须固定新问题、完整旧集、容量和权限一起验证，不在已观察题集上调参后宣称泛化已通过。本人模型与真实账号投递继续依用户选择延后。

## 真实讲义验证（2026-10-22）

使用已有 `knowledge/ICPC/博弈论1-讲义.md`，原文件未改。首次入库前固定四个问题及完整原文证据，分别询问普通 Nim、巴什博弈、反 Nim 全一特例、Sequential Nim；使用真实本地 CPU BGE-M3 和原 topK=5。原投影的证据排名依次为缺失 / 3 / 5 / 1。普通 Nim 的“问题”和“提示”是相邻同级小节，切片后只含题意的片段占据前列，核心结论未进入前五。

投影 v3 只对紧邻的同父级“问题 / 提示”（或 Question / Hint）补充最多 1000 字的原问题上下文，供提示片段生成向量。正文、标题路径、切片数量 / 次序、chunkId / chunkHash 和返回引用不变；TXT、PDF、视频及其他 Markdown 原向量输入保持原字节。章节层级、插入小节（包括空小节）都会断开关联，不跨章节取问题。旧 READY v2 仍可读，管理员显式重建同版本时才升级，原计数相同也不能略过；没有自动重建现有资料或新 SQL。

隔离真实模型比较后，四问完整证据排名为 **4 / 1 / 5 / 1**，引用身份与原文逐项一致。没有把条件改为命中部分文字，没有提高 topK 或重写标签。原 v1 / v2 / v3 / v4 共 212 问再次测量，88 个切片的模型输入、所有冻结资料 / 题目及各组质量指标与原线上 dense 基线相同。这里是无退步验证：v2 原 heldout recall@3 仍为 **23/26（88.46%）**，低于未修改的 90% 门槛，原评测运行器保留失败退出，不能称所有门槛通过；`context-assessment.json` 单独记录改善范围与既有缺口。普通 Nim 结论仍排第 4，不声称首条精确命中或泛化完成。

工作区相关单测实际 XML 194 项通过；准备部署的稳定 dense 源码另跑 171 项通过，二者不相加。早期临时目录权限、缺测试依赖和错误测试路径的失败结果保留，不计通过。复用已有完整本地测试环境后才得到以上结果，没有下载新模型。工作区 index.py 仍含上一轮未采用的词级实验，不能把整个目录直接当作本次发布产物。

本次从已运行稳定镜像 `sha256:6d4240ae840b54f0962116223ffb6cb6f233fe5e8e69c9a2dc47d8537b5320fe` 派生，只 COPY 稳定代码上的本次 index.py；构建无网络。新镜像 `sha256:863a5adff0638403af15c136c6d35168235a5030dc88fa7ff8c4ae525c8f919e` 已用于本地 RAG / 初始化容器。实际 index.py SHA256=`43cecfaabfa1718d86411273e7ec412e91bd200ab1c0c32e9d502f08dec7c385`，与隔离评测版本一致；词级模块与模式仍未部署 / 开放。后端、JudgeServer、资料卷和身份卷保留。

真实网站流程使用临时管理员上传 ADMIN 讲义，经常驻 worker / 真实向量入库后显式发布；四问均核对文档 / 版本 / 原文、引用 ID / URL，普通账号按指定 ID 也不能检索或阅读。没有 API 替代、伪设 INDEXED、聊天模型或外站请求。结束删除本轮临时资料并等待撤销任务，清理测试账号 / 元数据，向量点位恢复；原资料文件哈希保持。

首轮 HTTP 通过，但人工截图发现原文打开后停在页底；增加实际可见性断言，等待过渡结束后仍失败，保留失败报告。原因是读者在页面缩放过渡中提前定位；前端在根节点过渡结束后按原权限 / 请求代次重新定位。最终实际浏览器的引用标题位于视口 top≈120px，原文问题与提示可见，2 图已人工核对。Node 22 生产构建通过；原隔离知识库页面回归实际 19 组 / 12 图通过，错误为 0。受控页面回归与真实上传流程分别报告，不混作真实向量或生成质量。

证据在 `data/hoj-real-materials-20261022`：冻结 lecture-goldens、原失败 / 最终真实流程、上下文比较及四组完整 JSON、context-assessment、两份通过 XML、构建 / 部署日志、浏览器报告 / 图和最终 audit。`evaluate-context.py` 的整轮退出 1 来自保留的 v2 既有门槛失败；四组均 executionComplete / cleaned=true。`assess-context.py` 检查逐模型输入和旧基线指标，不能替代绝对门槛。

整本教材导入仍未完成：已有数论 PDF 为 178 页、约 27 MiB、约 22 万字，超过现有 8 MiB / 20 万字限制，200 页上限未超。本轮仅只读检查元数据 / 抽取 / 渲染，未修改原件、放宽产品限制或入库这本书。限制用于本地资源控制，与模型额度无关。完整真实教材、真实模型逐结论支持 / 无依据拒答、真实课堂识别质量等仍需后续验证，用户模型与源站本人 AC / WA 沿用延后决定。

## 整本数论 PDF 验证：2026-10-22，容量通过、检索未验收

本轮沿用原数论 PDF，原件 SHA-256 为 `f81a32204e310b44580b46ea856b3b9d24b7be72d13ddf5c9dc5b36daa9a715b`。产品的 PDF 上限现为 32 MiB / 200 页 / 30 万码点，普通文本仍 20 万；后端、前端、HMAC 请求边界、原件保存 / 下载与数据库列共同调整，具体资源契约见 hoj-integration.md。保持 512 MiB / 20 CPU 秒 / 25 墙钟秒解析子进程限制；PDF 向量入库独立允许 20 分钟、后台租约 21 分钟，普通操作没有扩大期限。

原 pypdf 在物理第 7 页产生空字符，未删除字符强行接纳。PDFium 替代提取能完成但丢失 `==` 等代码运算符，人工对照原图后弃用。复用原项目 PyMuPDF 并固定 1.28.0，在同一 Linux 资源限制下约 2.253 秒提取 178 页、221,894 码点，正文哈希 `0e7b41254a18db2b0230c82fde873866bf94eab874d9690943f001a501b48442`。检查的代码运算符保留，但部分数学公式仍缺失，部分正文与代码阅读顺序不同；本轮不能证明整本公式完整或生成答案正确。

四个问题、完整证据及允许物理页在首次检索前冻结，`book-goldens.json` SHA-256 为 `df9f543a17f49063676b1a5dc1dc04f8b76fbff86d0a07709b03978437127dcc`。不改题、不换证据、不放宽 topK=5。下面是实际管理页文件上传、真实常驻 worker / 本地 BGE / 磁盘 Qdrant / MySQL 的结果，没有 API mock、伪造入库状态或模型 / 源站请求。

| 问题 | 首轮，327 片段 | 章节上下文，367 片段 |
| --- | --- | --- |
| 试除法 is_prime 模板的循环与整除判断 | 两条证据均未进前五 | 两条证据均未进前五 |
| 埃氏筛与线性筛复杂度比较 | 未进前五 | 第 2 |
| 拓展欧几里得的算法思想 | 第 1 | 第 1 |
| 整除分块右边界更新 | 第 1 | 第 2 |

两轮分别约 422 / 623 秒完成入库，均 attempts=1；原件 / 正文哈希、178 物理页、显式发布、ADMIN 范围和普通用户拒绝均核对。检索范围为已发布允许版本，片段必须逐字属于原物理页。两轮最终 ingested=true、validated=false、cleaned=true，所有本轮资料 / 向量 / 临时账号已清理；第二轮在检索门槛失败后未执行阅读器验证，不把上传截图当作引用定位通过。最终报告在 `data/hoj-wholebook-20261022/live-first` 与 `live-context`；旧 progress 文件不证明任务仍运行。

章节投影 revision=4 只将原 PDF 明确编号章节作为内部向量上下文，跨页继承；引用仍是物理页、原文切片，不把上下文伪装成引用。原 212 问对应的 88 个片段、原文 / 路径和实际向量输入逐项不变，retrieve AST 未变。这是兼容性证据，没有把它称作重新测量了 212 个问题；原 v2 绝对召回门槛仍失败。新的代码上下文候选在本机离线模型中对同一组问题比较，尚未部署，也不是新的未见样本。

容量阶段 Python 209 项、章节阶段 212 项、Java 118 项实际 XML 均无失败 / 跳过；Java 包含真实 MySQL RR 的 20 MiB 原件和 PDF 租约 / 接管 / 旧 token 隔离。原主库 source_data 已从 mediumblob 无损扩大为 longblob，并重复迁移两轮；临时新库 97 表、22 个可重放迁移两轮通过并清理。Java 8 镜像与 Node 22 生产构建通过，原页面 19 组 / 12 图回归通过。第二轮实际后端为 d11eb753…、RAG 为 b452ded4… / dense。

未采用的词级实现已移至 `src/learning_os/hoj/experiments/`，默认 Dockerfile 构建上下文排除该目录，生产入口仅 dense / hybrid，Compose 固定 dense；实验评测显式引用实验模块，拒绝结果保留。14 个生产模块与第二轮实际镜像源码逐字节一致。隔离后的全量主机测试实际为 474 项中 473 通过、1 失败；这是 Windows Python 的 path.stat 与 fstat 对 ctime 含义不同，首个未修改文件即可误报。仅在 Windows 测试夹具将路径侧也改为真实描述符元数据，生产五字段校验保持；补同长度、恢复修改时间的真实变更回归后，相关 45 项通过、失败 / 错误 / 跳过 0。后续全量重跑停在 HTTP TestClient 夹具并中止，不能记为完整通过，也不能用主机结果代替 Linux 文件语义。

默认 Dockerfile 的独立标签重建先因沙盒无法读取 buildx 锁失败；提升权限后，自动审批返回审核额度不足，新构建和容器检查没有执行。此限制属于工具审批，与开源模型、PDF 字数或模型 API 额度无关。当前没有再部署镜像或 Git 提交；默认源码构建与 Linux 复核仍待执行，不把已有派生镜像运行当作默认构建通过。证据保留原失败日志、source-final.xml、source-portable-focused.xml、experiment-isolation.json、pdf-context-compatibility.json、两轮真实产品报告和离线候选日志。

## 代码标识符补召回：2026-10-22，本地通过、待部署验证

先保留两个失败实验：从原文补函数签名的方案仍为 [0,0] / [2] / [1] / [2]；将长正文与代码分开后有 382 个片段，仍为 [0,0] / [3] / [1] / [2]。后者没有解决代码问题且使另一问后移，均未加入生产源码。报告 code-context-probe / code-split-probe 保留实际 BGE-M3 的固定原文、问题、证据及 topK=5；使用缓存的耗时不是全量重新入库耗时。

当前方案保留原 367 片段与向量输入，只在普通语义候选完全缺少问题明确写出的下划线标识符时，使用现有授权 FTS 补至多一条原文到最后一位。FTS 候选不能以标题、大小写折叠或名称前后缀冒充正文精确命中；前四条保持，不移除其他已存在名称的唯一证据。完整允许版本 / namespace / READY 先于截断，候选仍从 Qdrant 取真实向量和片段，cosine 不伪装成关键词置信度；最终墓碑复核保留。

verify-code-book.py 使用前面同一实际模型、包版本和精确输入字节缓存的 float32 数值向量，在新的真实 SQLite / Qdrant 中正常 ingest / retrieve / delete。没有重编模型向量、HTTP 替代或主库 / 在线索引访问，也不是第三轮管理页上传。修复前重现 [0,0] / [2] / [1] / [2]，修复后为 **[5,5] / [2] / [1] / [2]**；四问全部完整证据前五命中，每问原前四条完全一致。原文、版本 / 哈希、物理页、chunkHash 全部核对；空授权不读模型向量、删除后不可检索、两份隔离索引清空，validated / cleaned=true。新片段是原物理第 7 页的既有片段，不改固定题或证据。

相关 XML code-identifiers.xml 实际 57 项、code-lexical.xml 实际 14 项，均无失败 / 错误 / 跳过，合计 71 个不同用例；包含既有 PDF / 视频、关键词及新代码补召回。覆盖真实 cosine 与来源、普通 / 已命中 / 无匹配查询、大小写 / token 边界 / 标题、多个名称覆盖、授权排除、空授权、候选读取期间另一 adapter 撤回、错误 namespace，以及既有 hybrid 行为。此前全量 HTTP TestClient 的中止与 Linux 待验证状态保留，不能把此子集称作全量测试通过。

replay-code-compatibility.py 核对四份已接受 dense 报告的冻结 cases 哈希、文档正文哈希、原文与 88 个实际向量输入，再将 212 个归档 top5 交给新增守卫回放，真实 SQLite 处理缺失名称查询。202 问没有这种名称、9 问已出现全部名称；唯一缺失 pop_back 的问题在允许资料中没有字面来源，FTS 无匹配，保留原结果。212 问排序 / 分数均不变，FTS 仅调用 1 次、候选向量读取 0 次。报告明确 semanticRemeasured=false：这是回放兼容性证据，不是重新编码并测量 212 问，原 v2 召回门槛依旧失败，也不证明任意新问题的回答质量。

该阶段新改动在 index.py / lexical.py，向量代次仍为 4，其他未采用实验继续隔离。正常 Docker 源码构建、Linux 校验、切换 RAG 后第三轮真实管理页 / 引用定位均未执行，原因是工具自动审批审核额度不足。已准备 data/hoj-wholebook-20261022/resume-source-check.ps1，语法已检查；该脚本仅构建独立标签、比较源码并验证 Linux 视频文件身份，不重启现有服务。用户随后选择工具恢复后由 Codex 执行；它不等于整本产品验收或发布完成。

## 长入库期间的检索可用性：2026-10-22，源码完成、待容器验证

先用真实 SQLite / Qdrant 与 Event 暂停编码，复现旧实现将入库和检索全程放在同一个存储锁下：即使检索另一份已 READY 的资料，线程也停在 retrieve 的锁入口。整本书此前实际入库 422 / 623 秒，共享 HTTP 两个容量槽又允许两个入库占满，不能把空授权健康探测成功当成真实检索可用。原始复现和堆栈保留在 data/hoj-wholebook-20261022/ingest-availability-probe.json。

当前变更增加 runtime.py：同一个 Qdrant client 的多个 adapter 共用存储锁和串行入库通道，同一个 embedder 共用推理调度；存储锁仅保护本地状态 / 点位操作，不覆盖模型计算。文档按原序每八个片段编码，查询在批次之间优先执行；有文档排队时最多连续执行四个查询，避免入库饥饿。所有锁 / 模型队列等待遵守原绝对截止时间，模型调用仍串行，正在计算的批次不能被中途抢占。检索在编码前取得 READY 三元组，编码后及返回前重新核对持久状态；入库每批排队前、真正编码前和发布前核对不可变元组及 WRITING，删除不等待模型，也不能被迟到入库复活。HTTP 先取得独立的单个 ingest 槽，再取得共享容量，第二个入库立即 429，为检索 / 撤销保留容量。提取 PDF / 视频的独立配额保持。

子代理只读审查发现排队前校验存在窗口：查询占用模型时删除文档，已排队批次仍会编码一次。新增确定顺序用例在修复前实际失败，queued-before.xml / log 保留；补实际编码前复核后进入最终全量测试。并发用例覆盖 8/8/1 三批间的有效授权查询、无模型并发、跨 adapter 立即撤销 / 无迟到点、跨 adapter 重试仅编码一次、不同模型共享存储序列化、HTTP 第二入库拒绝但查询正常及容量释放，以及推理公平性 / 截止 / 异常释放。没有通过 sleep 推断并发顺序。

主机 TestClient 的原全量挂起另有独立根因：最小 FastAPI 程序在业务代码之前，卡在 Windows Python 3.12 的 socketpair TCP 回环 accept。25 秒 faulthandler 已保存实际堆栈；本机 Python 未提供 AF_UNIX。仅为该测试进程准备基于内存 Event 的 SelectorEventLoop，任何 socket reader / writer 注册均立即拒绝，应用 / ASGI / 测试断言不变，不修改生产启动或全局 Python。框架 smoke 约 1.2 秒通过，旧代码补召回源码全量 487 项通过；当前可用性修复后最终 **500 个实际 XML testcase 全部通过，失败 / 错误 / 跳过 0，约 45.43 秒**。既有 Starlette TestClient 弃用提示保留。这是主机单元 / ASGI 契约验证，不是实际 TCP、Linux 或常驻容器验证；原默认 loop 的失败日志仍保留。

verify-real-book.py 已以离线实际 BGE-M3、全书原 367 片段和预先冻结的四问完成新批次路径复核，使用独立临时 SQLite / 内存 Qdrant，不访问业务库 / 线上索引，不调用用户模型。模型冷加载约 28.5 秒，实际入库约 **664.6 秒**，共 45 批八段及最后一批七段；不使用向量缓存跳过编码。另一个已 READY 的独立资料在首批计算时收到检索请求，约 **3.578 秒**返回正确允许资料，此时整本书仍 WRITING；模型调用无并发。该单次请求的测试截止为 120 秒，实际完成早于产品 Java 的 25 秒检索期限，但没有证明任意批次 / 长问题 / 并发负载的 25 秒响应承诺。此轮目标是可用性，不据不同环境 / 缓存状态的耗时比较宣称入库提速。

四问完整证据排名仍为 **[5,5] / [2] / [1] / [2]**，逐条核对原文、原物理页、完整版本及 chunkHash；空授权不调用模型，删除后不可检索，隔离点位清空，validated / cleaned=true。367 个实际新编码向量与原离线实验采用相同八段分组的缓存基线逐项比较，最大绝对差 0、最小 cosine=1；这不是与已清理的旧常驻容器整批编码向量作比较，也不是第三轮真实管理页上传。固定四问已参与诊断，不能据此推断未见资料 / 问题质量、公式完整或生成答案正确。

原四份冻结资料的 88 个片段向量输入逐项相同，每份文档最多 2–3 段，小于新批次边界，编码调用分组也不变；切片 / 授权范围 / 补召回方法 AST 与本轮前一致。这是兼容性核对，不是新的 212 问语义测量。证据归档在 data/hoj-rag-availability-20261022：actual-book.json / log、local-checkpoint.json / local-audit-final.log，以及 python-final.xml / log。最终用例计数以实际 XML 的 500 项为准，local-checkpoint 核对实现哈希、实际模型报告和冻结数据；goalComplete=false。

独立 Linux 可用性 smoke 已加入恢复脚本，与原 Linux 视频身份检查一起在未来构建镜像中执行；只挂载只读验证脚本，使用容器 /tmp 和内存 Qdrant，不挂业务数据 / 模型。验证查询在三批入库间完成、同模型不并发、同 client 多 adapter 共享归属、编码时撤销 / 无迟到点。脚本逻辑先在 Windows 主机通过，明确 linuxBehaviorVerified=false；首轮主机失败是沙盒默认临时目录无法打开 SQLite / 清理，保留日志，随后仅把主机夹具临时目录放到可写证据目录。恢复 PowerShell 语法再次通过，Docker / Linux 尚未执行。

用户已选择“工具恢复后，由 Codex 继续执行”。自动审批此前以审核额度不足拒绝执行新的 Docker 构建和检查，这不是模型额度，也不是判定动作危险；本轮未重试绕过，未重启现有服务。恢复后先执行独立源码镜像 / Linux 检查，再更新 RAG 并重跑真实管理端上传 / 发布 / 原文定位。公式提取、v2 原召回不足、本人模型质量及用户验收仍未完成，没有 Git 提交。

## 默认镜像恢复与排序候选否决：2026-10-22，历史执行记录（接续结果见下节）

接续时默认沙盒仍不能读取 Docker 命名管道；正常提升权限的只读 docker version 这次实际成功，服务端 29.0.1，自动审批已恢复。随后已启动原 resume-source-check.ps1，使用正常 Dockerfile.hoj 构建独立 source-verified 标签；不是再次复制派生运行镜像。真实构建已完成基础镜像 / PyTorch 下载，继续安装其余依赖。四份 Compose 的 config --quiet 实际通过，未输出环境或密钥。更新前重新检查：现有 RAG b452ded4… / healthy，后端 d11eb753…，其余原服务仍运行；这些只证明检查时的状态，不是新镜像验收。

已准备只更新 RAG 的 deploy-verified-source.ps1：要求源码比较和两个真实 Linux smoke 通过、当前服务仍是核对过的旧镜像，才重标 local 并使用原四份 Compose --no-deps / --no-build / --pull never 重建 RAG。保留所有原 data / identity / media 卷，不改数据库或重启后端 / JudgeServer，更新后检查实际镜像和四个生产文件哈希。该部署脚本已做 PowerShell 语法检查，尚未执行，不能提前记为更新完成。真实整本管理页测试补可选 availability 模式：先真实入库 / 发布一个独立 PUBLIC 资料，在全书 job RUNNING 时让普通用户通过本站非空检索，并检查书仍未完成；最终清理范围仍限本轮随机 created_by。脚本用实际 Node 24 语法检查通过，浏览器 / 实际执行待镜像完成。

构建等待期间完成一个独立排序实验。固定算法为保留 dense 的第一条，余下四位按既有 hybrid RRF 顺序去重填充；不调权重、问题、证据、分组、topK 或 90% 门槛，也未改生产源码 / 配置。四集采用当前源码、实际本机 BGE，重新编码原 88 个片段和全部 dense / hybrid 查询；报告包含源码、数据和模型配置哈希，使用临时真实 SQLite / 内存 Qdrant。全书四问另用前阶段实际数值缓存验证，明确不称重新编码。约 158.75 秒完成、executionComplete / cleaned=true。

| 固定题集 | 当前 dense heldout recall@3 | 候选 | 原前三证据退步数 |
| --- | ---: | ---: | ---: |
| v1 | 27/28 | 28/28 | 0 |
| v2 | 23/26 | 25/26 | 0 |
| 多格式 v3 | 19/20 | 20/20 | 0 |
| 复杂场景 v4 | 12/12 | 9/12 | 4 |

v4 的 window-tie 2→5、rollback-empty-union 3→5、gap-comparison [3,1]→[4,1]、duplicate-comparison [2,1]→[4,1]，原成功路径明显退步。该候选虽修复 v2 的 bridge / SAM 漏检，仍 **candidateGatePassed=false、noTopThreeLoss=false，已否决、不部署**。全书四问不退步和 v1/v2 历史重放的改善不能替代跨题集门槛。原 v2 问题正文证据全部存在，失败原因与后续实验继续保留；无依据题候选率仍未解决。

新实验与日志为 data/hoj-rag-availability-20261022/probe-anchored-hybrid.py、anchored-hybrid-probe.json / log；Docker 构建和 Linux 恢复日志仍在 data/hoj-wholebook-20261022。所有改动未提交 Git，完整七项目标继续进行。

## Linux 元数据碰撞与正常源码部署：2026-10-23，本地复验通过、待用户验收

正常 Dockerfile 首轮独立镜像 93480387… 成功构建，50 个源码文件与工作区一致，experiments 目录排除。但原 Linux 视频身份 smoke 失败：112 字节文件同长度改写并恢复 mtime 后，返回了成功回执。实际隔离容器再测 10 次，十次内容 SHA 均改变，而 dev / inode / size / mtime_ns / ctime_ns 五字段完全一致，旧实现十次全部接纳。此结果否定“只修 Windows 夹具即可证明 Linux 正确”，也说明此前主机 500 项不足。失败镜像没有部署。内核文档说明同一计时周期内的 inode 时间戳可能相同；这里的直接证据仍为本机实际元数据及字节变化，而非对所有文件系统行为的推断。[Linux 内核时间戳说明](https://www.kernel.org/doc/html/latest/filesystems/multigrain-ts.html)

修复移除源文件已验证缓存及模型版本的元数据缓存豁免，保留五字段、regular file、O_NOFOLLOW、路径与描述符绑定检查。源文件每次进入和成功返回前都按声明长度完整做 SHA256，先定位 0、每 MiB 检查截止、额外读取至多 1 字节排除增长，并在交给 worker 前恢复偏移；模型在每次转写前后计算完整内容版本。新测试只冻结 ctime 项以确定重现已观察的碰撞，其余真实读写 / inode / mtime 不变：旧实现实际 5 失败 / 2 通过，修复后相关 66 项与完整 **507 项**均通过；完整主机 XML 失败 / 错误 / 跳过 0，约 49.09 秒。覆盖冷 / 热调用期间改写、缓存后请求前已改写且 worker 不得执行、worker 将 FD 移至 EOF、路径换 inode、模型前后版本变化。未将前后哈希称为任意中途改写后完全恢复（ABA）的不可变快照保证；后台资产仍须按现有只写一次规则保存。

第二轮正常 Dockerfile 镜像 **767e8d2b…** 成功构建，50 个文件再次一致；扩展 Linux smoke 通过内容 / 模型修订 / FD 偏移和清理，原入库并发 / 撤销 Linux smoke 也通过。热 tmpfs 的 16 MiB 合成文件 guard 约 17–19 ms（两次全量读取、无 ffmpeg / ASR），大媒体和冷盘延迟仍需单独验证，不把它外推为 1 GiB 资源承诺。原失败构建与 Linux 日志复制留档在 data/hoj-video-integrity-20261023，未覆盖失败证据；新构建日志仍在前阶段恢复目录。

通过检查后，已执行带前置 guard 的 deploy-verified-source.ps1，仅将常驻 RAG 从 b452ded4… 更新为 767e8d2b…，健康与 index / runtime / service / lexical / video_extract 五个文件哈希核对通过。原 data / identity / media 卷保留，后端 d11eb753… 和 JudgeServer 未重建，没有 SQL 迁移或 Git 操作。该部署事实及完整镜像号在 ../data/hoj-rag-availability-20261022/source-deployment.json。

真实原创视频完整复验已通过：同一个原 80 秒中英合成视频，实际管理端上传、持久 worker / 本地 Whisper 形成四段字幕、BGE 入库、显式发布、范围读取 / 私有授权、时间引用、校订版本与可见性 / 正式及虚拟比赛限制均执行。实际浏览器跟随引用到 64.72 秒并播放，HttpOnly 范围 Cookie 核对；携带向量重启 RAG 后 healthy、原件 / 向量保留且不重新入库。validated / cleaned=true，原题与队列计数恢复，两个截图已人工查看，无 API mock、聊天模型或源站调用。报告在 data/hoj-video-integrity-20261023/live-video/knowledge-video-local-integration.json。这个合成视频不是未见课堂视频准确率或 1 GiB 介质性能证明。重新运行原 Linux 十次诊断，9 次实际五字段碰撞，0 次改写被接纳，见 linux-identity-diagnostic-after.log。

第三轮原 27 MiB 整本 PDF 真实管理端运行在 live-book：一次入库 542 秒，178 页 / 367 片段，四问完整证据排名 [5,5] / [2] / [1] / [2]，第 7 页代码与定位检查已通过；最后失败于验证脚本让 PDF 分支继续执行普通 Markdown / TXT 的默认“前缀”断言，取到的是 PDF 第 1 页正文。该轮 validated=false / retrievalValidated=true / cleaned=true，原报告保留，不回写为通过。

knowledge-live-browser.cjs 改为 PDF / 视频 / 普通文本互斥分支，PDF 分支显式核对版本和 format，仍保留 178 页、引用第 7 页、完整代码字符串、导航下方实际可见性和浏览器错误断言。Node 24 语法检查通过后在新目录 live-book-final 重新运行完整实际流程，没有复用旧入库结果或弱化四个冻结问题。

最终实际运行 validated / ingested / retrievalValidated / cleaned 全为 true：管理端真实上传原件、原件 / 正文哈希和 ADMIN 隔离、显式发布、持久 worker / BGE 一次入库 **556 秒 / 367 片段**；四问仍为 **[5,5] / [2] / [1] / [2]**，返回逐项核对原物理页 / 实际字符串 / 完整版本和 chunkHash。实际浏览器跟随试除法引用进入第 7 页，178 页 DOM 与该页 `if(x % i == 0)` 原文字串核对通过，页顶在导航下方约 **120px**；搜索与原文两图已人工查看。原文截图是该页上半部，不是代码所在位置的单独截图；截图也确认了公式缺失和提取顺序差异。该真实容器结果不再由前阶段主机结果替代。

同轮另一个实际先发布的 PUBLIC 资料在整本入库期间，通过普通用户与真实后端非空检索约 **708 ms** 返回；查询前后均按 namespace / documentId / version / contentHash 只读核对 Python WRITING，且 MySQL job 仍 RUNNING，modelUsed=false。此单次延迟不代表任意负载或长文档的响应上限。

最终 audit-release.py 实际通过：运行镜像 / 50 文件比对、原后端镜像、507 个实际 testcase、Linux 检查、两份真实业务报告与原 PDF 哈希均复核；七张资料 / 视频表及向量点清空至测试前状态，原业务计数相同，7 服务运行 / 6 healthy，前后端可用，已取消的平时 Rating 端点仍返回业务 400。原身份初始化容器保留，未读写服务 Key。报告为 data/hoj-video-integrity-20261023/release-final-state.json，completeGoal=false、gitCommitted=false、userAcceptance=pending。

原生文字 PDF 目前确实不能按页校订，OCR_REVIEWED 可复用流程已梳理，但 NATIVE_REVIEWED 尚未实现，不能称已解决缺失公式。下一步先复用现有不可变版本 / 物理页校验为原生 PDF 增加显式管理员校订，再继续原 v2 检索缺口；真实用户模型与本人源站 AC / WA 按原决定延后。冷盘 / 大视频读取成本、未见课堂资料准确率及逐结论生成 / 无依据拒答仍未验收。

## 原生 PDF 按页校订：2026-10-23，已本地更新、待用户验收

整本 PDF 的前阶段实际截图确认原生提取仍缺公式，且只有 OCR 来源能编辑。现复用同一物理页校订界面 / 服务端校验与不可变版本，为 NATIVE / NATIVE_REVIEWED 也开放显式管理员校订。新增 reviewedPdf，旧 reviewedOcr 在 OCR 家族保持兼容；来源家族取自服务端存储，不能靠浏览器 ocr 值改家族。校订创建 NATIVE_REVIEWED 或 OCR_REVIEWED；原件 / 自动提取版本 / 已发布指针保留，索引完成后仍须单独发布。完整公开契约在 HOJ 的 docs/docs/ai/08-data-model-and-api.md。

KnowledgePdfUpload 保存服务端确定的来源模式，原始上传不能复制浏览器 review 标记。reuse 不再把继承的校订状态写回调用 DTO，避免同一元数据对象重复执行时误把忽略的正文当校订。原 native / OCR / OCR_REVIEWED 三组旧指纹以固定 SHA 值回归，新中性 OCR 请求与旧标记重放产生同一回执；原生新模式单独纳入校订正文哈希。物理页 / 空页、规范分页、UTF-8 / NUL / 30 万码点边界、CRLF、原件哈希和管理员权限维持。

实际 8 套件 **124 testcase 全通过，失败 / 错误 / 跳过 0**；Java 8 生产构建与 Node 22 生产构建分别成功。主机 JDK 21 仅覆盖使用已缓存 Lombok 1.18.46，生产仍用原 Lombok。首个 Maven 调用参数拆分和第二次本地未缓存版本读取失败的日志保留，均未执行测试；其后的 48 项不是最终全量，最终计数只取 host-final-xml。没有新增数据库列、锁 / 租约、RAG / JudgeServer 代码或模型依赖。

最终受控 Vue 回归 **26 组 / 19 图**通过，无异常 / 外站请求；两种 PDF 的真实 input 事件、中性标记、来源标签、标题编辑不传正文、手机深色和原视频 / 权限行为覆盖。整本现有 178 页 / 221,894 码点的正文另由受控 API 装入编辑器，检查 178 个物理页输入，第 7 页修改后页数保留，放弃草稿没有新版本。该项是浏览器容量证据，不是实际整本新版本 / BGE 入库；没有将它与前阶段 556 秒实际全书流程混称。

本轮真实业务分别在 live-native 和 live-ocr：原创三页文字 PDF、原创三页扫描件各自真实上传 / 解析或 OCR / BGE / worker / 发布；随后管理员在实际页面逐页输入校订，保存为 v3，新模式分别 NATIVE_REVIEWED / OCR_REVIEWED，旧发布仍 v2。核对原件字节相同、v2 正文未改、草稿用户不可读、同 nonce 重放及变化 409；OCR 用旧 reviewedOcr 重放新页面命令成功。索引后显式发布，旧用户链接失效，新正文可检索，实际浏览器跟随引用到校订第 1 页。ADMIN / 正式和虚拟参赛限制继续验证。两个报告 validated / cleaned=true，无 API mock / 聊天模型 / 源站调用；实际校订和阅读截图已查看。

仅后端 d11eb753… → **a7660e51…**，RAG 767e8d2b… 与 JudgeServer ac1ae3d7… 保持，全部原卷及已完成身份初始化容器保留。首次部署动作已经成功，但等待不存在的 Docker Health 字段而以失败终止；deploy-backend-first.ps1 / deployment.log 保留。修正后通过 VerifyOnly 检查同一个实际新容器的 running / API 200，没有第二次重建。默认沙盒启动器之后两次只读检查遇文件占用错误，授权环境完成读取与审计，未据此改产品权限或服务限制。

audit-final.py 实际通过：运行后端镜像与构建 local 标签相同，RAG 50 个 Python 文件仍与源码一致，7 服务运行 / 其余 6 healthy，前后端可用；原业务计数 3 / 8 / 9 / 6 / 0 / 1 / 0 / 0 不变，七张资料 / 视频表和向量点回到 0，本轮账号清空。两个 Git HEAD / HOJ cdutetc-ui 分支保留，无 SQL 迁移、Git 提交或用户模型调用。最终证据入口为 data/hoj-native-pdf-20261023/final-state.json，completeGoal=false。

尚未自动补回或人工逐页校正整本教材的所有公式，校订后的全文仍由管理员对照原件负责。本轮也没有解决原 v2 23/26 的 90% 召回门槛、资料不足判定或真实模型逐结论依据质量；原先否决的 RRF / 词级排序不重新启用。下一步继续这些检索和依据质量缺口，真实本人模型与源站 AC / WA 保持用户选择的延后，大媒体 / 冷盘和未见课堂准确率不宣称验收完成。
