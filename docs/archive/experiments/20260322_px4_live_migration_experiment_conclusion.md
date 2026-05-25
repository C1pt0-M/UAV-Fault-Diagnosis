# PX4 Live 杩佺Щ瀹為獙闃舵鎬х粨璁猴紙2026-03-22锛?
## 鏂囨。鐩殑

鏈枃妗ｇ敤浜庢眹鎬诲綋鍓嶉樁娈靛洿缁?**PX4 SITL live telemetry -> 鍦ㄧ嚎浜屽垎绫绘晠闅滄娴?* 鎵€鍋氱殑鍏抽敭楠岃瘉銆佸疄楠岀粨鏋滀笌闃舵鎬х粨璁恒€?
鐩爣涓嶆槸缁欏嚭鏈€缁堟柟妗堬紝鑰屾槸鏄庣‘鍥炵瓟浠ヤ笅闂锛?
1. 褰撳墠椤圭洰鐨勬柊 telemetry live 閾捐矾鏄惁宸茬粡璺戦€氥€?2. 褰撳墠 PX4 live 鏍锋湰鏄惁婊¤冻鍩烘湰杈撳叆闂ㄧ銆?3. 鐜版湁妯″瀷锛圧fly 妯″瀷 / 鏃?PX4 妯″瀷锛夋槸鍚﹀彲浠ョ洿鎺ラ浂鏍锋湰杩佺Щ銆?4. 濡傛灉涓嶈兘锛屾渶鍙兘鐨勫師鍥犳槸浠€涔堛€?5. 褰撳墠闃舵鏄惁杩樺€煎緱缁х画鎶曞叆闆舵牱鏈瘯楠岋紝杩樻槸搴旇浆鍚戦€傞厤璁粌銆?
---

## 1. 宸插畬鎴愮殑鍏抽敭宸ヤ綔

### 1.1 鏂?telemetry live 閾捐矾宸叉墦閫?
褰撳墠椤圭洰涓殑瀹炴椂閲囬泦閾捐矾鐢变袱閮ㄥ垎缁勬垚锛?
- `telemetry.collector`
- `telemetry.receiver`

宸查獙璇侊細

- 鍙湪 PX4 SITL + QGroundControl 鍦烘櫙涓嬮噰闆?live telemetry
- 閲囬泦鐜浣跨敤锛歚/home/jack/miniconda3/envs/fault-dataset`
- 鎺ㄨ崘 MAVSDK 鐩戝惉鍦板潃锛歚udpin://0.0.0.0:14540`
- 鏂伴摼璺兘鎴愬姛鐢熸垚 live CSV 鏂囦欢

### 1.2 鏂伴摼璺瓧娈靛凡琛ラ綈骞堕獙璇?
鐩稿浜庢棫 live 鏍锋湰閾捐矾锛屽綋鍓嶉」鐩柊閾捐矾宸茬‘璁ゅ彲閲囧埌浠ヤ笅瀛楁锛?
- `baro_pressure_pa`
- `mag_x`
- `mag_y`
- `mag_z`
- `rates_sp_roll`
- `rates_sp_pitch`
- `rates_sp_yaw`
- `thrust_z`

鍚屾椂宸插鍔犺ˉ鍏呮祦鍙娴嬫爣蹇楋細

- `pymav_highres_imu_seen`
- `pymav_attitude_target_seen`
- `pymav_actuator_control_seen`

缁撹锛?
- 鏂?live 閾捐矾宸叉槑鏄句紭浜庢棫鐗?`telemetry_live` 鐨勫熀纭€瀛楁瀹屾暣搴︺€?
### 1.3 Phase A 杈撳叆闂ㄧ鑴氭湰宸茶惤鍦?
宸叉柊澧烇細

- `scripts/phase_a_input_gate.py`

鍙緭鍑猴細

- `phase_a_summary.json`
- `phase_a_feature_coverage.csv`
- `phase_a_risk_fields.csv`

骞跺凡閫氳繃鐩稿叧娴嬭瘯銆?
---

## 2. 褰撳墠 PX4 live 姝ｅ父鏍锋湰鐘舵€?
褰撳墠鐢ㄤ簬楠岃瘉鐨?live 鏍锋湰涓猴細

- `results/telemetry_new_live_probe.csv`

### 2.1 Phase A 缁撴灉

宸查獙璇佺粨鏋滃寘鎷細

- `rows = 4870`
- `time_axis_valid = 1`
- `valid_window_ratio = 1.0`
- `mapped_ratio 鈮?0.611`
- `derived_ratio 鈮?0.258`
- `default_ratio 鈮?0.131`

缁撹锛?
- 褰撳墠 live 鏍锋湰鍦ㄨ緭鍏ュ眰闈㈠睘浜?*鍚堟牸鏍锋湰**銆?- 鏃堕棿杞淬€侀噰鏍风巼銆佹粦绐楁瀯閫犲拰瀛楁瀹屾暣搴﹀凡缁忚冻澶熻繘鍏ユā鍨嬭瘎浼般€?
### 2.2 鐢垫満杈撳嚭宸茬‘璁や笉鏄叏 0

瀵?`motor1..4` 鐨勭粺璁＄粨鏋滄樉绀猴細

- 闈為浂甯ф暟绾?`4532 / 4870`
- 鏁板€艰寖鍥寸害 `0.0 ~ 0.82~0.86`

缁撹锛?
- 褰撳墠 live 鏍锋湰涓嶆槸鈥滅┖杈撳嚭鏍锋湰鈥?- 涔熶笉鏄€滅數鏈烘案杩滀笉杞€濈殑浼甯告牱鏈?
---

## 3. 闆舵牱鏈縼绉诲疄楠岀粨鏋?
### 3.1 Rfly 璁粌鐨勫湪绾夸簩鍒嗙被妯″瀷

妯″瀷璺緞锛?
- `assets/models/online_detector_cnn1d_rfly`

鍦ㄥ綋鍓?PX4 live 姝ｅ父鏍锋湰涓婄殑缁撴灉锛?
- `mean_fault_prob = 1.0`
- `max_fault_prob = 1.0`
- `p_fault_prob_ge_05 = 1.0`
- `n_events = 1`

缁撹锛?
- **瀹屽叏澶辫触**
- 姝ｅ父鏍锋湰琚叏绋嬪垽瀹氫负 fault

### 3.2 鏃?PX4 鍦ㄧ嚎浜屽垎绫绘ā鍨?
妯″瀷璺緞锛?
- `models_old/assets/models/online_detector_CNN`

鍦ㄥ悓涓€ PX4 live 姝ｅ父鏍锋湰涓婄殑缁撴灉锛?
- `mean_fault_prob = 1.0`
- `max_fault_prob = 1.0`
- `p_fault_prob_ge_05 = 1.0`
- `n_events = 1`

缁撹锛?
- **鍚屾牱瀹屽叏澶辫触**
- 鍗充娇浣跨敤鏃х殑 PX4 鏁版嵁璁粌妯″瀷锛屽綋鍓?live 鏍锋湰浠嶇劧鍏ㄧ▼琚垽 fault

### 3.3 warm-up 瑁佸壀鍚庨噸璇勪及

鎴戜滑瀵瑰綋鍓?live 鏍锋湰鍋氫簡棰濆瑁佸壀锛?
- 浠?`motor1..4` 杩炵画 50 甯ч兘澶т簬闃堝€间负璧风偣
- 瑁佹帀鏂囦欢澶?warm-up 娈?
寰楀埌锛?
- `trim_start_idx = 338`
- `rows_trimmed = 4532`

鐒跺悗鍒嗗埆閲嶈窇锛?
- Rfly 妯″瀷
- 鏃?PX4 妯″瀷

缁撴灉浠嶇劧閮芥槸锛?
- `mean_fault_prob = 1.0`
- `p_fault_prob_ge_05 = 1.0`

缁撹锛?
- 褰撳墠闂**涓嶆槸 warm-up 娈甸€犳垚鐨勫亣闃虫€?*銆?
---

## 4. 鍏煎瀛愮壒寰侀泦瀹為獙缁撴灉

### 4.1 绗竴鐗堥珮椋庨櫓瀛楁鍓旈櫎鏂规

鎴戜滑灏濊瘯灞忚斀浠ヤ笅楂橀闄╁瓧娈碉細

- `actuator_out_*`
- `actuator_mean/std/effort`
- `motor_cross_*`
- `baro_temp`
- `baro_pressure`
- `mag_*`
- `vel_x`
- `vel_y`
- `actuator_ctrl_*`

缁撴灉锛?
- `mean_fault_prob = 1.0`
- `p_fault_prob_ge_05 = 1.0`

### 4.2 鏋佺畝 `Core-IMU` 瀛愰泦

鍙繚鐣欙細

- `gyro_*`
- `accel_*`

缁撴灉浠嶇劧鏄細

- `mean_fault_prob = 1.0`
- `p_fault_prob_ge_05 = 1.0`

缁撹锛?
- 闂**涓嶅彧鏄煇浜涘潖瀛楁姹℃煋**
- 涔熶笉鏄垹鎺?actuator/baro/mag 灏辫兘鏁戝洖鏉?
---

## 5. RflyMAD 瀹樻柟鏁版嵁闆嗙悊瑙ｄ笌鏈湴瀵规瘮缁撹

### 5.1 RflyMAD 瀹樻柟缁勬垚

瀹樼綉鏄庣‘璇存槑锛孯flyMAD 鍖呭惈锛?
- `SIL`
- `HIL`
- `Real`

骞朵笖鏁版嵁鏉ユ簮鍖呮嫭锛?
- `ULog`
- `Telemetry log`
- `Ground Truth Data`
- `BAG`

缁撹锛?
- RflyMAD 涓嶆槸鍗曚竴浠跨湡鍩熸暟鎹泦銆?
### 5.2 涓?RflyMAD SIL No Fault 鐨勫姣?
鎴戜滑閲嶆柊閫夊彇鐪熸鐨?`SIL + No Fault` 鏍锋湰杩涜瀵规瘮锛?
- `Case_1010000025.csv`

瀵规瘮褰撳墠 PX4 SITL live normal 鍚庡彂鐜帮細

楂樺亸绉诲瓧娈靛寘鎷細

- `mag_x`
- `mag_y`
- `mag_z`
- `baro_temp`
- `actuator_out_1..4`
- `vel_x`
- `vel_y`

缁撹锛?
- **鍗充娇鍙姣?RflyMAD 鐨?SIL No Fault锛孭X4 SITL normal 涔熶笌鍏跺樊寮傛樉钁椼€?*
- 鎵€浠ュ綋鍓嶅け璐ヤ笉鑳界畝鍗曞綊鍥犱簬 `HIL/REAL` 娣峰叆銆?
---

## 6. 褰撳墠鏈€鍙俊鐨勯樁娈垫€у垽鏂?
鍩轰簬浠ヤ笂鎵€鏈夊疄楠岋紝褰撳墠闃舵鏈€鍙俊鐨勫垽鏂涓嬶細

### 鍒ゆ柇 1

褰撳墠 PX4 live telemetry 鏂伴摼璺凡缁忔墦閫氾紝涓旀牱鏈湰韬川閲忚冻澶熻繘鍏ヨ瘎浼般€?
### 鍒ゆ柇 2

鐜版湁鍦ㄧ嚎浜屽垎绫绘ā鍨嬫棤璁烘槸锛?
- Rfly 璁粌妯″瀷
- 鏃?PX4 璁粌妯″瀷

鍦ㄥ綋鍓?PX4 live normal 涓婇兘**瀹屽叏闆舵牱鏈け璐?*銆?
### 鍒ゆ柇 3

澶辫触鍘熷洜宸茬粡涓嶈兘鍐嶄富瑕佸綊鍥犱负锛?
- 瀛楁缂哄け
- 鐢垫満鍏?0
- warm-up 娈?- 涓埆楂橀闄╁瓧娈?
褰撳墠鏇村彲淇＄殑鍘熷洜鏄細

- **璁粌鍩熷拰閮ㄧ讲鍩熸暣浣撲笉鍏煎**
- 鍖呮嫭浣嗕笉闄愪簬锛?  - 浠跨湡骞冲彴宸紓
  - 棰勫鐞嗛摼宸紓
  - 褰掍竴鍖栫粺璁″樊寮?  - 鐘舵€佸垎甯冨樊寮?
### 鍒ゆ柇 4

褰撳墠缁х画杩藉姞闆舵牱鏈瘯楠岀殑杈归檯浠峰€煎凡缁忓緢浣庛€?
---

## 7. 褰撳墠鏈€鍚堢悊鐨勪笅涓€姝ユ柟鍚?
褰撳墠涓嶅缓璁啀缁х画鎶曞叆澶ч噺鏃堕棿鍋氾細

- 鏇村闆舵牱鏈ā鍨嬪垏鎹?- 鏇村瀛楁灞忚斀璇曢獙
- 鏇村 warm-up 瑁佸壀璇曢獙

### 鏇村悎鐞嗙殑涓や釜鏂瑰悜

#### 鏂瑰悜 A锛氳繘鍏ラ€傞厤璁粌闃舵

鍗筹細

- 鍩轰簬褰撳墠 live schema
- 鏀堕泦 PX4 SITL normal/fault 鏍锋湰
- 閲嶆柊寤虹珛 live-compatible 鐨勮缁?寰皟鏁版嵁閾?
#### 鏂瑰悜 B锛氬鏋滅煭鏈熷彧姹傚彲鐢ㄦ€э紝鍒欐殏缂撳垏鎹㈡暟鎹摼

鍗筹細

- 鏆傛椂涓嶄緷璧栧綋鍓?live 鏁版嵁鐩存帴鍋氬湪绾挎晠闅滄娴?- 鍏堢淮鎸佸師鏈夎缁?璇勪及浣撶郴
- 鎶婂綋鍓嶅疄楠岀粨璁轰綔涓衡€滆法鍩熻縼绉诲け璐モ€濈殑闃舵鎬х粨鏋滀繚瀛?
---

## 8. 涓€鍙ヨ瘽鎬荤粨

褰撳墠闃舵宸茬粡鍙互鏄庣‘寰楀嚭锛?
> **PX4 live 鏂伴摼璺凡缁忛噰閫氾紝浣嗙幇鏈夊湪绾夸簩鍒嗙被妯″瀷锛堟棤璁?Rfly 妯″瀷杩樻槸鏃?PX4 妯″瀷锛夐兘鏃犳硶鍦ㄥ綋鍓?PX4 SITL normal 鏍锋湰涓婇浂鏍锋湰宸ヤ綔銆傞棶棰樼殑涓诲洜涓嶆槸閲囬泦閾捐矾锛岃€屾槸璁粌鍩熶笌閮ㄧ讲鍩熸暣浣撲笉鍏煎銆?*
