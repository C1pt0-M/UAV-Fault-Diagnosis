# 鍦ㄧ嚎妫€娴嬩唬鐮佸垎鏋愯褰曡崏绋匡紙鍩轰簬浠ｇ爜闃呰锛?
## 1. 鍩烘湰淇℃伅

- 瀹為獙鍚嶇О锛氬湪绾挎娴嬭緭鍏ュ悎鍚屼笌瀛楁閾捐矾浠ｇ爜瀹¤
- 瀹為獙鏃ユ湡锛?026-03-22
- 璁板綍浜猴細Claude Code
- 瀹為獙闃舵锛?  - [x] 杈撳叆鍏煎鎬у疄楠?  - [ ] 闆舵牱鏈縼绉诲疄楠?  - [ ] 鍏煎瀛愮壒寰侀泦瀹為獙
  - [ ] 瑙勫垯铻嶅悎瀹為獙
  - [ ] 鍑嗗疄鏃跺洖鏀惧疄楠?- 鍏宠仈鏂囨。锛?  - `/home/jack/UAV-Fault-Diagnosis/docs/cross_platform_experiment_record_template.md`
  - `/home/jack/UAV-Fault-Diagnosis/src/telemetry/schema_adapter.py`
  - `/home/jack/UAV-Fault-Diagnosis/src/telemetry/config.py`
  - `/home/jack/UAV-Fault-Diagnosis/src/telemetry/receiver.py`
  - `/home/jack/UAV-Fault-Diagnosis/src/online/realtime_detection_service.py`
  - `/home/jack/UAV-Fault-Diagnosis/src/online/fault_detector.py`
  - `/home/jack/UAV-Fault-Diagnosis/src/data/build_online_cache.py`
  - `/home/jack/UAV-Fault-Diagnosis/src/data/build_sequence_cache.py`
  - `/home/jack/UAV-Fault-Diagnosis/apps/dashboard/backend/app/services/telemetry_manager.py`

## 2. 鏈疆鐩爣

### 鏈疆瑕佸洖绛旂殑闂

- 褰撳墠鍦ㄧ嚎妫€娴嬫ā鍨嬬殑璁粌杈撳叆鍚堝悓鏄粈涔堛€?- PX4 live 閬ユ祴杩涘叆 detector 鍓嶆槸鍚﹀凡缁忓畬鎴?canonical 鍖栥€?- live 瀛楁銆乧anonical 瀛楁銆佽缁冨瓧娈典箣闂存槸鍚﹀瓨鍦ㄥ懡鍚嶆垨璇箟鏂銆?
### 鏈疆鍒ゅ畾鏍囧噯

- 鎴愬姛锛氳兘鏄庣‘鍒楀嚭璁粌 schema銆佸湪绾挎帹鐞?schema銆乴ive 閲囬泦 schema 涓庝富瑕侀闄╃偣銆?- 閮ㄥ垎鎴愬姛锛氳兘瀹氫綅涓婚摼璺紝浣嗕粛鏈夊叧閿繍琛屾椂琛屼负寰呭疄娴嬨€?- 澶辫触锛氭棤娉曚粠浠ｇ爜纭 detector 杈撳叆鍚堝悓銆?
---

## 3. 杈撳叆鏁版嵁涓庤繍琛岀幆澧?
### 3.1 璁粌渚ц緭鍏ュ悎鍚岋紙宸茬‘璁わ級

- `build_sequence_cache.py` 瀹氫箟 canonical schema锛歚SEQ_COLUMNS` 鍏?31 涓瓧娈点€?- 鍏朵腑鍩虹瀛楁 27 涓紝鏉ヨ嚜 `SEQ_COLUMN_ALIASES`锛涙淳鐢熷瓧娈?4 涓細
  - `actuator_mean`
  - `actuator_std`
  - `acc_norm`
  - `actuator_effort`
- `build_online_cache.py` 澶嶇敤 `_extract_channels()`锛屽洜姝ゅ湪绾挎ā鍨嬭缁冭緭鍏ヤ笌涓婅堪 canonical schema 缁戝畾銆?- `norm_stats.json` 鍐欏叆 `columns=SEQ_COLUMNS`锛屾帹鐞嗙鎸夎椤哄簭鍙栧€煎苟褰掍竴鍖栥€?
### 3.2 live 閬ユ祴閲囬泦鍚堝悓锛堝凡纭锛?
- `receiver.py` 鎸佷箙鍖栧垪涓猴細`[seq, time_s] + FEATURE_COLUMNS + AUX_COLUMNS`銆?- `FEATURE_COLUMNS` 鏄?PX4 live / supplement / state 娲剧敓鐨勬贩鍚?schema锛屼笉绛夊悓浜?`SEQ_COLUMNS`銆?- live 鎸佷箙鍖栧瓧娈典腑鍖呭惈锛?  - 鐩存帴 live 鍚嶏細`motor1..4`銆乣vx/vy/vz`銆乣baro_alt_meter`銆乣baro_temp`
  - 璁粌鏈洿鎺ヤ娇鐢ㄤ絾琚褰曠殑鎵╁睍瀛楁锛歚battery_*`銆乣rates_sp_*`銆乣thrust_z`銆乣esc_rpm_*`銆乣rate_err_*` 绛夈€?
### 3.3 鎺ㄧ悊渚ц緭鍏ュ悎鍚岋紙宸茬‘璁わ級

- `FaultDetector` 浠庢ā鍨嬬洰褰?`norm_stats.json` 璇诲彇 `columns`銆乣mean`銆乣std`銆?- 瀹為檯鎺ㄧ悊杈撳叆鐭╅樀 shape 涓?`(T, C)`锛屽綊涓€鍖栧悗閫佸叆妯″瀷锛岀獥鍙ｉ暱搴﹂粯璁?150銆?- detector 浼樺厛鎸?`row[col_name]` 鐩存帴鍙栧€硷紱鑻ュ瓧娈电己澶憋紝浠呭灏戦噺娲剧敓椤瑰仛琛ョ畻锛?  - `actuator_mean` / `actuator_std` / `actuator_effort`
  - `motor_cross_mean` / `motor_cross_std` / `motor_effort`
  - `acc_norm`
  - `rate_err_*`
  - 绋€鐤忓垪 `accel_vibration_metric` / `gyro_vibration_metric` 鍋?last-value forward fill
- detector **娌℃湁**鍐呭缓 PX4 live -> canonical 鐨勫埆鍚嶆槧灏勩€?
---

## 4. 瀛楁鍏煎鎬ц褰?
### 4.1 瀛楁瑕嗙洊鎽樿锛堟寜浠ｇ爜鑳藉姏涓婇檺锛?
- 妯″瀷 canonical 杈撳叆瀛楁鏁帮細31
- `schema_adapter` 鍙洿鎺ユ槧灏勭殑 canonical 鍩虹瀛楁鏁帮細27
- `schema_adapter` 鍥哄畾娲剧敓瀛楁鏁帮細4
- `time_s` / `seq` 浠呬綔涓烘椂搴忓厓鏁版嵁锛屼笉灞炰簬 `SEQ_COLUMNS`
- 杩愯鏃?`default` 瀛楁鏁帮細鍙栧喅浜?live 甯у疄闄呯己鍒楋紝褰撳墠鏈仛瀹炴祴缁熻

### 4.2 鍏抽敭瀛楁鏄犲皠

| canonical 瀛楁 | PX4 live 鏉ユ簮 |
|---|---|
| `actuator_out_1..4` | `motor1..4` |
| `actuator_ctrl_1..4` | `actuator_ctrl_1..4` |
| `baro_alt` | `baro_alt_meter` |
| `baro_temp` | `baro_temp` |
| `baro_pressure` | `baro_pressure_pa` |
| `vel_x/vel_y/vel_z` | `vx/vy/vz` |
| `att_q_*` | `q_*`锛岀己澶辨椂鐢?`roll/pitch/yaw` 鎺ㄥ |
| `mag_x/mag_y/mag_z` | `mag_x/mag_y/mag_z` |

### 4.3 楂橀闄╁瓧娈?
| 瀛楁/閾捐矾 | 闂绫诲瀷 | 椋庨櫓绛夌骇 | 璇存槑 |
|---|---|---|---|
| `RealtimeDetectionService -> FaultDetector.predict` | schema 鏂 | 楂?| 鐩存帴鎶?`TelemetryReceiver.window` 閫佸叆 detector锛屾湭鍏堣皟鐢?`adapt_px4_live_frame()` |
| `baro_alt` / `baro_pressure` / `vel_x` 绛?canonical 鍚?| 鍛藉悕涓嶄竴鑷?| 楂?| raw live 琛岄噷閫氬父鏄?`baro_alt_meter` / `baro_pressure_pa` / `vx`锛宒etector 涓嶄細鑷姩鍒悕杞崲 |
| `att_q_*` | 璇箟琛ラ綈渚濊禆 | 涓?| manager 璺緞鍙敱 Euler 鎺?quaternion锛況aw receiver 璺緞鑻ユ湭 canonicalize锛屽垯 detector 鐩存帴鎷夸笉鍒?`att_q_*` |
| `TARGET_HZ=50` 涓嬮噰鏍?| 鏃跺簭鍋囪 | 涓?| `build_online_cache.py` 浠呮帴鍙楄繎鏁存暟鍊嶇巼涓嬮噰鏍凤紝闈炴暣鏁板€嶇巼浼氱洿鎺ユ姏閿?|
| 璁粌/鎺ㄧ悊鍒楅『搴?| 寮鸿€﹀悎 | 涓?| detector 瀹屽叏渚濊禆 `norm_stats.json[columns]`锛屾ā鍨嬬紦瀛樺彉鏇翠細鐩存帴褰卞搷绾夸笂鍙栧€奸『搴?|

### 4.4 褰撳墠鍒ゆ柇

- [ ] 瀛楁鍏煎鎬ф暣浣撳彲鎺ュ彈
- [x] 瀛樺湪灏戦噺楂橀闄╁瓧娈碉紝浣嗕笉闃诲浠ｇ爜灞傜粨璁?- [ ] 楂橀闄╁瓧娈佃緝澶氾紝寤鸿鏆傚仠杩涘叆妯″瀷瀹為獙

澶囨敞锛氬綋鍓嶆渶绐佸嚭鐨勯棶棰樹笉鏄ā鍨嬬粨鏋勶紝鑰屾槸鈥渞aw receiver schema鈥濅笌鈥渃anonical training schema鈥濅箣闂村瓨鍦ㄥ弻璺緞琛屼负涓嶄竴鑷淬€?
---

## 5. 閲囨牱鐜囦笌绐楀彛绋冲畾鎬ц褰?
### 宸茬‘璁ゅ弬鏁?
- `window_size`: 150
- `stride`: 50
- `target_hz`: 50.0
- `fault_threshold`: 0.5
- `max_windows_per_file`: 8

### 浠ｇ爜灞傝瀵?
- `build_online_cache.py` 鍏堜笅閲囨牱鍐嶅垏绐楋紝绐楀彛鏍囩鎸?`fault-active` 鍗犳瘮鏄惁澶т簬 `0.5` 鍐冲畾銆?- `RealtimeDetectionService` 鍜?`TelemetryReceiver` 榛樿绐楀彛闀垮害涔熸槸 150锛岃缁?鎺ㄧ悊绐楀彛灏哄鏄竴鑷寸殑銆?- 浣?live 閲囨牱鐜囨槸鍚︾ǔ瀹氬湪璁粌鍒嗗竷闄勮繎锛屽綋鍓嶆湭鍋氬疄娴嬮獙璇併€?
### 寰呴獙璇侀」

- `[unverified]` live 瀹為檯 `dt_median` / `dt_p95`
- `[unverified]` live 閾捐矾鏈夋晥绐楀彛姣斾緥
- `[unverified]` 寮辩綉鎴?replay 涓嬬獥鍙ｆ柇瑁傞鐜?
---

## 6. 浠ｇ爜璺緞缁撹

### 6.1 鐩稿瀹夊叏鐨勯摼璺?
- `apps/dashboard/backend/app/services/telemetry_manager.py` 涓?`_on_backend_frame()`锛?  - PX4 璺緞浼氬厛璋冪敤 `adapt_px4_live_frame(frame)`
  - detector 鏀跺埌鐨勬槸 canonical frame / canonical window
  - 鍚屾椂璁板綍 `feature_coverage`
- 杩欐潯閾捐矾涓庤缁?schema 鏇翠竴鑷淬€?
### 6.2 褰撳墠鏈€鍗遍櫓鐨勯摼璺?
- `src/online/realtime_detection_service.py` 涓?`_on_frame()`锛?  - 鐩存帴鎵ц `self.detector.predict(list(window))`
  - `window` 鏉ユ簮浜?`TelemetryReceiver`
  - `TelemetryReceiver` 淇濆瓨鐨勬槸 live schema锛屼笉鏄?canonical schema
- 鍥犳璇ヨ矾寰勪笅锛宒etector 瀵?canonical 鍒楃殑澶ч噺璇诲彇浼氶€€鍖栨垚 0 鎴栫己鐪佽ˉ鍊笺€?
### 6.3 褰卞搷鑼冨洿

- 褰卞搷鐨勪笉鏄?CSV 鎸佷箙鍖栨湰韬紝鑰屾槸 standalone realtime detection service 鐨勮緭鍏ヨ涔夊畬鏁存€с€?- backend manager 璺緞涓?standalone realtime service 璺緞鐩墠骞朵笉绛変环銆?
---

## 7. 缁撹

### 鏈疆缁撹鍒嗙骇

- [ ] 鍙洿鎺ヨ縼绉?- [x] 鍙儴鍒嗚縼绉?- [ ] 鏆備笉鍙洿鎺ヨ縼绉?
### 缁撹璇存槑

- 宸茬‘璁よ缁冪 canonical schema銆佹帹鐞嗙鍒楅『搴忓拰 live 閲囬泦 schema 涓夎€呭苟涓嶅ぉ鐒剁浉鍚屻€?- 宸茬‘璁?backend `telemetry_manager` 璺緞閫氳繃 `schema_adapter` 鍋氫簡 canonical 鍖栵紱standalone `realtime_detection_service` 璺緞娌℃湁鐪嬪埌鍚岀瓑澶勭悊銆?- 宸茬‘璁?detector 鑳借ˉ灏戦噺娲剧敓瀛楁锛屼絾涓嶈兘鏇夸唬瀹屾暣鐨?live -> canonical 瀛楁鏄犲皠銆?
### 褰撳墠鏈€鍙兘鐨勯棶棰樺眰绾?
- [x] 瀛楁鏄犲皠闂
- [x] 璇箟涓嶄竴鑷撮棶棰?- [ ] 缁熻鍒嗗竷鍋忕Щ闂
- [x] 閲囨牱鐜?/ 鏃跺簭闂
- [ ] 妯″瀷娉涘寲鑳藉姏闂
- [ ] 瑙勫垯灞傜己澶遍棶棰?
### 涓嬩竴姝ュ缓璁紙浠呭垪寰呴獙璇侊紝涓嶈涓哄凡瀹屾垚锛?
- 鐢ㄧ湡瀹?PX4 live 鏍锋湰鍚屾椂璧?`telemetry_manager` 涓?`realtime_detection_service`锛屽姣?detector 杈撳叆鐭╅樀闈為浂鐜囥€?- 瀵?standalone realtime service 澧炲姞 feature coverage 瑙傛祴锛岀‘璁?`default` 鍒楀崰姣斻€?- 瀵?live CSV 缁熻瀹為檯閲囨牱鐜囷紝楠岃瘉鏄惁婊¤冻 `target_hz=50` 鐨勮缁冨亣璁俱€?