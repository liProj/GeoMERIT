# GeoMERIT 宸ョ▼缁撴瀯涓庝唬鐮佺畝浠?
鏈粨搴撴槸 GeoMERIT 璁烘枃鐨?GitHub 寮曠敤鐗堜唬鐮佸寘锛岀敤浜庢敮鎾戣鏂囦腑鐨勬柟娉曞疄鐜般€佸疄楠屽鐜板拰鍥捐〃鐢熸垚銆備粨搴撳彧鍖呭惈浠ｇ爜銆侀厤缃€佽交閲忕粨鏋滄憳瑕佸拰璁烘枃鍥捐〃璧勪骇锛涗笉鍖呭惈 FORCE 2020 鍘熷 LAS 鏁版嵁銆佸ぇ鍨嬬壒寰佽〃銆佸畬鏁?OOF 姒傜巼鐭╅樀鍜屾ā鍨嬬紦瀛樸€?
## 鐩綍缁撴瀯

```text
GeoMERIT-GitHub-Release
鈹溾攢鈹€ geomerit/              # 鏍稿績鏂规硶浠ｇ爜
鈹溾攢鈹€ scripts/               # 瀹為獙杩愯鑴氭湰
鈹溾攢鈹€ configs/               # 鏁版嵁銆佺壒寰併€佹ā鍨嬨€佹儵缃氱煩闃甸厤缃?鈹溾攢鈹€ results/               # 杞婚噺瀹為獙缁撴灉鎽樿
鈹溾攢鈹€ paper/                 # 鏈€鏂拌鏂?LaTeX銆佸浘琛ㄥ拰浣滃浘鑴氭湰
鈹溾攢鈹€ requirements.txt       # Python 渚濊禆
鈹溾攢鈹€ README.md              # GitHub 棣栭〉璇存槑
鈹斺攢鈹€ CITATION.cff           # GitHub 寮曠敤鍏冩暟鎹?```

## 鏍稿績浠ｇ爜妯″潡

| 鏂囦欢 | 浣滅敤 |
|---|---|
| `geomerit/io_las.py` | 璇诲彇 FORCE 2020 LAS 鏂囦欢鍜?NPD 杈呭姪 Excel 琛?|
| `geomerit/features.py` | 鏋勯€犵己澶辨劅鐭ャ€佸紓甯告劅鐭ャ€佺獥鍙ｄ笂涓嬫枃鍜屾搴︾壒寰?|
| `geomerit/labels.py` | 12 绫诲博鎬ф爣绛炬槧灏勩€佸湴璐ㄧ矖绫绘槧灏勩€侀暱灏剧被瀹氫箟 |
| `geomerit/weights.py` | 绫诲埆鏉冮噸銆佽竟鐣屾潈閲嶃€佹爣绛剧疆淇″害鏉冮噸 |
| `geomerit/models.py` | LightGBM/XGBoost/CatBoost 铻嶅悎銆佺矖鍒扮粏鍒嗙被銆佸熬绫讳笓瀹?|
| `geomerit/decode.py` | logit adjustment銆丅ayes-risk penalty 瑙ｇ爜銆侀棬鎺х瓥鐣?|
| `geomerit/metrics.py` | Weighted F1銆丮acro F1銆丅oundary F1銆丳enalty銆乀ail F1 |
| `geomerit/cv.py` | 鎸変簳鍒嗙粍鐨?GroupKFold 楠岃瘉宸ュ叿 |

## 瀹為獙鑴氭湰

| 鑴氭湰 | 浣滅敤 |
|---|---|
| `scripts/00_build_dataset.py` | 浠庡師濮?LAS 鍜?Excel 鏂囦欢鏋勫缓鐗瑰緛琛?|
| `scripts/01_train.py` | 杩愯 10 鎶?GroupKFold 璁粌 |
| `scripts/02_predict_decode.py` | 浣跨敤鎯╃綒鐭╅樀杩涜 Bayes-risk 瑙ｇ爜骞惰瘎浼?|
| `scripts/03_ablation.py` | 娑堣瀺瀹為獙鍏ュ彛 |
| `scripts/04_georacs_oof.py` | GeoRACS/OOF 鍚庡鐞嗕笌璇婃柇瀹為獙 |
| `scripts/04_make_figures.py` | 鍩虹璁烘枃鍥捐〃鐢熸垚 |
| `scripts/05_make_paper_figures.py` | 璁烘枃椋庢牸鍥捐〃鐢熸垚 |

## 璁烘枃寮曠敤寤鸿

璁烘枃涓彲鍐欙細

> The source code, configuration files, lightweight result summaries, and figure-generation scripts are publicly available at: `https://github.com/liProj/GeoMERIT`.

濡傛灉闇€瑕佹洿姝ｅ紡锛屽彲浠ュ啓锛?
> We release the GeoMERIT implementation as a paper-reference repository, including the core Python package, experiment entry points, model configurations, decoding settings, lightweight result summaries, and manuscript figure-generation scripts.

## 涓嶄笂浼犵殑澶ф枃浠?
浠ヤ笅鍐呭娌℃湁鏀惧叆 GitHub 浠撳簱锛?
- FORCE 2020 鍘熷 LAS 鏁版嵁锛?- `feature_table.parquet` 瀹屾暣鐗瑰緛琛紱
- 瀹屾暣 `decode_report.csv` 閫愯 OOF 棰勬祴锛?- `.npy` 姒傜巼鐭╅樀銆佹绱㈠厛楠屻€乻tacking 杈撳嚭锛?- 杩滅▼鏈嶅姟鍣ㄥ畬鏁村揩鐓э紱
- 浠讳綍 API key銆佹湇鍔″櫒瀵嗙爜鎴栨湰鍦扮幆澧冨嚟鎹€?

