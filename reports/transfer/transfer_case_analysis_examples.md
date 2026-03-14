# Transfer Case Analysis Examples

## Suggested Subsection Draft

Table below lists 8 representative cases selected from current formal variants. We intentionally include both success and failure examples. For `ja_en`, we highlight boundary failures to avoid overstating the method. For `FBDB15K` and `FBYG15K`, we highlight samples where the transfer-enhanced model recovers the correct target from severe baseline ranking errors.

| Dataset | Type | Source Entity | GT | Baseline | Ours | Brief Conclusion |
|---|---|---|---|---:|---:|---|
| ja_en | failure | Inspiration is DEAD | Inspiration Is Dead | 0 | 5 | 该样本说明 ja_en 上仍存在细粒度近邻歧义，当前方法的增益并不覆盖全部困难样本。 |
| ja_en | failure | Microsoft Windows 10 Mobile | Windows 10 Mobile | 1 | 1 | 该样本说明 ja_en 上仍存在细粒度近邻歧义，当前方法的增益并不覆盖全部困难样本。 |
| FBDB15K | success | 0gt 0v | Post-bop | 2008 | 0 | 该样本说明方法在 FBDB15K 上能把被 baseline 严重误排的目标实体拉回 top-1。 |
| FBDB15K | success | 02qjv1p (tv.tv program.number of seasons, tv.tv program.air date of final episode, tv.tv program.episode running time) | The Pacific (miniseries) | 655 | 0 | 该样本说明方法在 FBDB15K 上能把被 baseline 严重误排的目标实体拉回 top-1。 |
| FBYG15K | success | 05css  (film.film.initial release date) | Saboteur (film) | 66 | 0 | 该样本说明方法在 FBYG15K 上能把被 baseline 严重误排的目标实体拉回 top-1。 |
| FBYG15K | success | 029kpy (location.geocode.longitude, topic server.population number, location.geocode.latitude) | Amritsar | 30 | 0 | 该样本说明方法在 FBYG15K 上能把被 baseline 严重误排的目标实体拉回 top-1。 |
| ja_en | failure | ファット・マイク | Fat Mike | 2 | 13 | 该样本说明 ja_en 上仍存在细粒度近邻歧义，当前方法的增益并不覆盖全部困难样本。 |
| FBDB15K | success | 02p97 (computer.programming language.introduced) | JavaScript | 274 | 0 | 该样本说明方法在 FBDB15K 上能把被 baseline 严重误排的目标实体拉回 top-1。 |

## Per-Case Notes

### Case 1: ja_en / failure / idx=3275

- Source entity: `Inspiration is DEAD` (id=2114)
- Ground truth target: `Inspiration Is Dead` (id=12614)
- Baseline prediction: top-1=`Inspiration Is Dead`, rank of GT=`0`
- Our method prediction: top-1=`Mada Minu Ashita ni`, top-2=`Nimrod (album)`, rank of GT=`5`
- Possible reason: baseline 已命中 top-1，但方法将注意力转移到语义相近的音乐实体，rank 从 0 升至 5。
- Mechanism interpretation: 目标域适应整体有效，但在同域音乐/作品实体的细粒度边界上仍可能发生过度迁移。
- One-line takeaway: 该样本说明 ja_en 上仍存在细粒度近邻歧义，当前方法的增益并不覆盖全部困难样本。

### Case 2: ja_en / failure / idx=1201

- Source entity: `Microsoft Windows 10 Mobile` (id=6140)
- Ground truth target: `Windows 10 Mobile` (id=16640)
- Baseline prediction: top-1=`Windows 8`, rank of GT=`1`
- Our method prediction: top-1=`Windows 8`, top-2=`Windows 10 Mobile`, rank of GT=`1`
- Possible reason: baseline 与方法都把正确答案排到第 2 位附近，说明该样本更像是细粒度歧义而非完全失效。
- Mechanism interpretation: 同系列版本实体高度相似，表面特征和模态证据不足以完全分离近邻候选。
- One-line takeaway: 该样本说明 ja_en 上仍存在细粒度近邻歧义，当前方法的增益并不覆盖全部困难样本。

### Case 3: FBDB15K / success / idx=2283

- Source entity: `0gt 0v` (id=8091)
- Ground truth target: `Post-bop` (id=15953)
- Baseline prediction: top-1=`Funk rock`, rank of GT=`2008`
- Our method prediction: top-1=`Post-bop`, top-2=`Funk rock`, rank of GT=`0`
- Possible reason: baseline 的正确答案排位很靠后（rank=2008），方法恢复到 top-1，属于典型的大幅纠错样本。
- Mechanism interpretation: 伪种子质量控制与保守迁移共同降低了跨图谱噪声，帮助模型恢复正确目标实体。
- One-line takeaway: 该样本说明方法在 FBDB15K 上能把被 baseline 严重误排的目标实体拉回 top-1。

### Case 4: FBDB15K / success / idx=7880

- Source entity: `02qjv1p` (id=10849)
- Source hint: `tv.tv program.number of seasons, tv.tv program.air date of final episode, tv.tv program.episode running time`
- Ground truth target: `The Pacific (miniseries)` (id=21876)
- Baseline prediction: top-1=`Epic Movie`, rank of GT=`655`
- Our method prediction: top-1=`The Pacific (miniseries)`, top-2=`A Streetcar Named Desire (1951 film)`, rank of GT=`0`
- Possible reason: baseline 的正确答案排位很靠后（rank=655），方法恢复到 top-1，属于典型的大幅纠错样本。
- Mechanism interpretation: 伪种子质量控制与保守迁移共同降低了跨图谱噪声，帮助模型恢复正确目标实体。
- One-line takeaway: 该样本说明方法在 FBDB15K 上能把被 baseline 严重误排的目标实体拉回 top-1。

### Case 5: FBYG15K / success / idx=4851

- Source entity: `05css ` (id=7944)
- Source hint: `film.film.initial release date`
- Ground truth target: `Saboteur (film)` (id=28237)
- Baseline prediction: top-1=`Iron Man (2008 film)`, rank of GT=`66`
- Our method prediction: top-1=`Saboteur (film)`, top-2=`Iron Man (2008 film)`, rank of GT=`0`
- Possible reason: baseline 的正确答案排位很靠后（rank=66），方法恢复到 top-1，属于典型的大幅纠错样本。
- Mechanism interpretation: strict source 与 staged fresh-IL 提升了候选质量，使正确目标在噪声较大的跨图谱场景中重新排到首位。
- One-line takeaway: 该样本说明方法在 FBYG15K 上能把被 baseline 严重误排的目标实体拉回 top-1。

### Case 6: FBYG15K / success / idx=2903

- Source entity: `029kpy` (id=9995)
- Source hint: `location.geocode.longitude, topic server.population number, location.geocode.latitude`
- Ground truth target: `Amritsar` (id=25839)
- Baseline prediction: top-1=`Sacramento, California`, rank of GT=`30`
- Our method prediction: top-1=`Amritsar`, top-2=`University of Mumbai`, rank of GT=`0`
- Possible reason: baseline 的正确答案排位很靠后（rank=30），方法恢复到 top-1，属于典型的大幅纠错样本。
- Mechanism interpretation: strict source 与 staged fresh-IL 提升了候选质量，使正确目标在噪声较大的跨图谱场景中重新排到首位。
- One-line takeaway: 该样本说明方法在 FBYG15K 上能把被 baseline 严重误排的目标实体拉回 top-1。

### Case 7: ja_en / failure / idx=9563

- Source entity: `ファット・マイク` (id=6200)
- Ground truth target: `Fat Mike` (id=16700)
- Baseline prediction: top-1=`Thundercat (musician)`, rank of GT=`2`
- Our method prediction: top-1=`Krizz Kaliko`, top-2=`Thundercat (musician)`, rank of GT=`13`
- Possible reason: baseline 与方法都把正确答案排到第 2 位附近，说明该样本更像是细粒度歧义而非完全失效。
- Mechanism interpretation: 同系列版本实体高度相似，表面特征和模态证据不足以完全分离近邻候选。
- One-line takeaway: 该样本说明 ja_en 上仍存在细粒度近邻歧义，当前方法的增益并不覆盖全部困难样本。

### Case 8: FBDB15K / success / idx=1959

- Source entity: `02p97` (id=6030)
- Source hint: `computer.programming language.introduced`
- Ground truth target: `JavaScript` (id=15853)
- Baseline prediction: top-1=`Atromitos F.C.`, rank of GT=`274`
- Our method prediction: top-1=`JavaScript`, top-2=`Lisp (programming language)`, rank of GT=`0`
- Possible reason: baseline 的正确答案排位很靠后（rank=274），方法恢复到 top-1，属于典型的大幅纠错样本。
- Mechanism interpretation: 伪种子质量控制与保守迁移共同降低了跨图谱噪声，帮助模型恢复正确目标实体。
- One-line takeaway: 该样本说明方法在 FBDB15K 上能把被 baseline 严重误排的目标实体拉回 top-1。

## Thesis-Ready Paragraph

The case study further shows that the proposed transfer-enhanced framework brings different types of evidence across datasets. On `FBDB15K` and `FBYG15K`, the main benefit is large-rank recovery: samples that were ranked hundreds or even thousands of positions away by the baseline are restored to top-1 after introducing target-domain adaptation, stricter pseudo-label control, and more conservative transfer loading. In contrast, `ja_en` still contains fine-grained boundary failures, especially among highly similar music, media, and product entities. Therefore, the current evidence supports that the proposed mechanisms improve transfer robustness and candidate quality, while also indicating that cross-lingual fine-grained ambiguity remains an open challenge.
