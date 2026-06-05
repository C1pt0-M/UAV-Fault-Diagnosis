# Dashboard 鍓嶅悗绔鎺ユ枃妗?
## 1. 鐩綍缁撴瀯

```text
dashboard/
  frontend/   # Vue3 + Element Plus + ECharts
  backend/    # Flask + SQLite + JWT + 寮傛浠诲姟
```

杩愯鏃剁洰褰曪紙鍚庣鑷姩鍒涘缓锛夛細

```text
apps/dashboard/backend/runtime/
  uploads/    # 鍘熷涓婁紶鏂囦欢锛圲LG/CSV锛?  derived/    # ULG prepare 鍚庣敓鎴愮殑 CSV
  runs/       # 鎺ㄧ悊浠诲姟杈撳嚭
  reports/    # 璇婃柇鎶ュ憡 PDF
  dashboard.db
```

## 2. 閴存潈鏈哄埗

- 鐧诲綍鎺ュ彛锛歚POST /api/auth/login`
- 璇锋眰澶达細`Authorization: Bearer <token>`
- 鎵€鏈変笟鍔℃帴鍙ｅ潎瑕佹眰鐧诲綍锛坄/health`銆乣/api/auth/*`銆乣/api/setup/*` 闄ゅ锛?
榛樿绠＄悊鍛橈紙棣栨鍚姩鑷姩鍒涘缓锛夛細

- 鐢ㄦ埛鍚嶏細`admin`
- 瀵嗙爜锛歚admin123`

## 3. 鏍稿績涓氬姟娴佺▼

### 3.1 鏃ュ織绠＄悊锛堣祫浜х鐞嗭級

1. 涓婁紶璧勪骇锛歚POST /api/assets/upload`锛坄multipart/form-data`锛屽瓧娈?`file`锛?2. 鏌ヨ璧勪骇锛歚GET /api/assets?kind=all|ulg|csv`
3. 涓嬭浇璧勪骇锛歚GET /api/assets/{asset_id}/download`
4. 鍒犻櫎璧勪骇锛歚DELETE /api/assets/{asset_id}`
5. ULG 杞崲锛歚POST /api/assets/{asset_id}/prepare`锛堝紓姝ヤ换鍔★級

璇存槑锛?
- CSV 涓婁紶鍚庤涓哄凡 prepared锛堝彲鐩存帴鐢ㄤ簬妫€娴?鍒嗘瀽锛夈€?- ULG 涓婁紶鍚庨渶鍏堟墽琛?prepare锛屾垚鍔熷悗 `meta.prepared_csv_path` 浼氬啓鍏ャ€?
### 3.2 鏁呴殰妫€娴嬶紙绂荤嚎浜斿垎绫伙級

1. 鏌ヨ妯″瀷锛歚GET /api/models`
2. 鍒涘缓浠诲姟锛歚POST /api/runs`锛岃姹備綋锛?
```json
{
  "task": "offline_5class",
  "asset_id": 12,
  "model": "lstm"
}
```

3. 杞浠诲姟锛歚GET /api/runs/{run_id}`
4. 鑾峰彇缁撴灉锛歚GET /api/runs/{run_id}/result`

浠诲姟鐘舵€侊細

- `queued` -> `running` -> `success|error`

### 3.3 鏁版嵁鍒嗘瀽

1. 瀛楁鍏冩暟鎹細`GET /api/analysis/fields?asset_id=...`
2. 鏃跺簭鏇茬嚎锛歚GET /api/analysis/timeseries?asset_id=...&fields=a,b,c&start_t=...&end_t=...&max_points=...`

`/api/analysis/fields` 杩斿洖鍖呭惈锛?
- `field_groups[]`
- `time`
- `numeric_only: true`锛堝綋鍓嶅彧鏀寔鏁板€煎瓧娈电粯鍥撅級
- `non_numeric_fields[]`锛堣杩囨护鐨勯潪鏁板€煎瓧娈碉紝濡?`flight_phase`锛?
`/api/analysis/timeseries` 杩斿洖鍖呭惈锛?
- `x[]`
- `series[]`
- `downsample_step`
- `n_rows_raw`
- `n_rows_out`

### 3.4 璇婃柇鎶ュ憡锛堢绾夸簲鍒嗙被锛?
1. 鐢熸垚鎶ュ憡锛歚POST /api/reports/diagnosis`锛堣姹備綋锛歚{"run_id": 123}`锛?2. 鎶ュ憡鍒楄〃锛歚GET /api/reports/diagnosis?asset_id=&label=&page=&per_page=`
3. 鎶ュ憡璇︽儏锛歚GET /api/reports/diagnosis/{id}`
4. 涓嬭浇 PDF锛歚GET /api/reports/diagnosis/{id}/pdf`
5. 鍒犻櫎鎶ュ憡锛歚DELETE /api/reports/diagnosis/{id}`

鐢熸垚闄愬埗锛?
- 浠呭厑璁?`offline_5class` 涓?`status=success` 鐨勪换鍔＄敓鎴愭姤鍛娿€?- 鍚屼竴涓?`run_id` 鍙細鍒涘缓涓€浠芥姤鍛婏紙閲嶅璇锋眰杩斿洖宸插瓨鍦ㄧ粨鏋滐級銆?
鎶ュ憡缁撴瀯鎽樿锛?
- 鍩烘湰淇℃伅锛堣祫浜с€佹ā鍨嬨€佹椂闂达級
- 璇婃柇缁撹锛坙abel/confidence锛?- TopK 姒傜巼
- 鍏抽敭鍙傛暟鎽樿锛堟潵鑷换鍔＄粨鏋滃揩鐓э級
- 寤鸿椤癸紙鎸夌被鍒槧灏勶級

## 4. 鍏抽敭鎺ュ彛娓呭崟

- `GET /api/overview`
- `GET /api/models`
- `POST /api/assets/upload`
- `GET /api/assets`
- `GET /api/assets/{asset_id}`
- `GET /api/assets/{asset_id}/download`
- `DELETE /api/assets/{asset_id}`
- `POST /api/assets/{asset_id}/prepare`
- `POST /api/runs`
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/result`
- `GET /api/analysis/fields`
- `GET /api/analysis/timeseries`
- `POST /api/reports/diagnosis`
- `GET /api/reports/diagnosis`
- `GET /api/reports/diagnosis/{id}`
- `GET /api/reports/diagnosis/{id}/pdf`
- `DELETE /api/reports/diagnosis/{id}`

## 5. 閿欒绾﹀畾

- 鏈櫥褰曪細`401`
- 璧勪骇鏈?prepare锛歚409`

```json
{
  "success": false,
  "message": "asset_not_prepared",
  "prepare_endpoint": "/api/assets/{id}/prepare"
}
```

- 鎶ュ憡鐢熸垚鏉′欢涓嶆弧瓒筹細`409`锛堜换鍔℃湭鎴愬姛锛夋垨 `400`锛堜换鍔＄被鍨嬮敊璇級
- 浠诲姟澶辫触锛歚status=error`锛屽苟鍦ㄤ换鍔″璞′腑甯?`error` / `error_detail`

## 6. 鍙岀幆澧冩ˉ鎺?
鍚庣閫氳繃瀛愯繘绋嬭皟鐢ㄥ彲閰嶇疆鐨?Python / 澶栭儴鑴氭湰锛?
- `ML_PYTHON_BIN=`锛氱暀绌烘椂鑷姩鍥為€€鍒板綋鍓嶅悗绔?Python / `python3` / `python`
- `TELEMETRY_PYTHON_BIN=`锛氱暀绌烘椂澶嶇敤 `ML_PYTHON_BIN`
- `PX4_ROOT=~/PX4-Autopilot`
- `FAULT_DATASET_ROOT=`锛氭寚鍚戝閮?`Fault Dataset` 鏍圭洰褰?- `ULG_PREPARE_SCRIPT=`锛氫紭鍏堢骇楂樹簬 `FAULT_DATASET_ROOT`

鐢ㄩ€旓細

- ULG -> CSV 杞崲锛坄build_fault_dataset.py`锛?- 绂荤嚎浜斿垎绫绘帹鐞嗭紙`apps/dashboard/backend/scripts/offline5_infer.py`锛?
璇存槑锛?
- `ULG_PREPARE_SCRIPT` 宸查厤缃椂锛屽悗绔洿鎺ヨ皟鐢ㄨ鑴氭湰
- `ULG_PREPARE_SCRIPT` 鐣欑┖鏃讹紝鎵嶄細灏濊瘯浠?`FAULT_DATASET_ROOT/scripts/tools/build_fault_dataset.py` 鎺ㄥ
- `MODEL_ROOT`銆乣OFFLINE_NORM_STATS`銆乣AI_KNOWLEDGE_DIR` 绛夌浉瀵硅矾寰勬寜浠撳簱鏍圭洰褰曡В鏋愶紝閫傚悎 WSL2 涓嬬洿鎺ュ鐢?
## 7. 鍚姩鏂瑰紡

榛樿鍚姩鑴氭湰鎸変互涓嬮『搴忓鎵捐В閲婂櫒锛?
- `DASHBOARD_BACKEND_PYTHON`
- `apps/dashboard/backend/.venv/bin/python`
- `apps/dashboard/backend/.venv/Scripts/python.exe`
- `python3`
- `python`

```bash
# 1) 鍚姩鍚庣
bash scripts/start_dashboard_backend.sh

# 2) 鍚姩鍓嶇
bash scripts/start_dashboard_frontend.sh

# 鎴栦竴閿惎鍔?bash scripts/start_dashboard.sh
```

濡傛灉浣犲湪 Windows 瀹夸富涓婏紝閫氳繃 WSL2 杩愯鏁村 Dashboard锛?
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard_wsl.ps1
```

鍓嶇榛樿锛歚http://localhost:5173`  
鍚庣榛樿锛歚http://localhost:5000`
