# 20260322 PX4-Rfly 璺ㄥ钩鍙板湪绾挎娴嬭繘灞曡褰?
## 1. 璁板綍鐩爣

杩欎唤鏂囨。鐢ㄤ簬鎸佺画璁板綍 PX4 涓?RflySim / RflyMAD 璺ㄥ钩鍙板湪绾挎娴嬪伐浣滅殑浠ｇ爜瀹¤缁撹銆佸綋鍓嶅垽鏂€佸緟楠岃瘉椤逛笌鍚庣画琛屽姩銆傚悗缁鏈夋柊杩涘睍锛岀洿鎺ュ湪姝ゆ枃妗ｈ拷鍔犳洿鏂般€?
**缁存姢绾﹀畾锛?* 鏈枃浠朵綔涓哄綋鍓嶈法骞冲彴鍦ㄧ嚎妫€娴嬬殑鍗曚竴鎸佺画璁板綍鍩虹嚎锛涢樁娈垫€ф眹鎶ョ鍙崟鐙敓鎴愶紝浣嗗凡楠岃瘉鐨勬柊缁撹搴斿洖鍐欏埌鏈枃浠躲€?
## 2. 褰撳墠鎬讳綋鍒ゆ柇

- 褰撳墠缁撹锛?*鍙儴鍒嗚縼绉伙紙Yellow锛?*銆?- 鏇村噯纭湴璇达細**鐜版湁鍦ㄧ嚎浜屽垎绫绘ā鍨嬪彲浠ュ皾璇曡縼绉诲埌 PX4 瀹炴椂鍦烘櫙锛屼絾鍓嶆彁鏄繀椤昏蛋 backend 涓凡缁?canonicalize 鐨勯摼璺紝涓嶈兘鐩存帴鎶?standalone realtime service 瑙嗕负褰撳墠瀹夊叏鍩虹嚎銆?*
- 褰撳墠鏈€涓昏椋庨櫓涓嶅湪妯″瀷缁撴瀯鏈韩锛岃€屽湪锛?  1. live schema 涓?canonical training schema 涓嶄竴鑷达紱
  2. 涓嶅悓杩愯閾捐矾瀵?schema 鐨勫鐞嗕笉涓€鑷达紱
  3. PX4 live 瀹為檯閲囨牱鐜囥€佽鐩栫巼銆佽涔変竴鑷存€у皻鏈疄娴嬨€?
## 3. 宸查€氳繃浠ｇ爜纭鐨勪簨瀹?
### 3.1 璁粌 / 鎺ㄧ悊鍚堝悓

- 鍦ㄧ嚎妯″瀷鐩綍锛歚/home/jack/UAV-Fault-Diagnosis/assets/models/online_detector_cnn1d_rfly/config.json`
- 宸茬‘璁ゆā鍨嬩换鍔′负 `online_binary`锛屼笉鏄洓鍒嗙被銆?- 宸茬‘璁よ緭鍏ョ獥鍙ｉ暱搴?`window_size=150`銆?- 宸茬‘璁よ緭鍏?schema 涓?`rfly_seq_columns_v1`銆?- 宸茬‘璁ょ壒寰佹暟涓?`31`銆?- 宸茬‘璁?detector 瀹為檯鎸?`norm_stats.json.columns` 鍙栧垪骞跺綊涓€鍖栥€?
### 3.2 canonical schema 鏉ユ簮

- canonical 璁粌瀛楁浜嬪疄鏉ユ簮锛歚/home/jack/UAV-Fault-Diagnosis/src/data/build_sequence_cache.py`
- 宸茬‘璁?`SEQ_COLUMNS = 27 涓熀纭€瀛楁 + 4 涓淳鐢熷瓧娈礰銆?- 4 涓浐瀹氭淳鐢熷瓧娈典负锛?  - `actuator_mean`
  - `actuator_std`
  - `acc_norm`
  - `actuator_effort`

### 3.3 PX4 live 鍒?canonical 鐨勯€傞厤鑳藉姏

- PX4 live 閫傞厤鍏ュ彛锛歚/home/jack/UAV-Fault-Diagnosis/src/telemetry/schema_adapter.py`
- 宸茬‘璁?`adapt_px4_live_frame()` 鑳芥妸 `motor1..4`銆乣vx/vy/vz`銆乣baro_alt_meter`銆乣baro_pressure_pa`銆乣q_*` 鎴?`roll/pitch/yaw` 绛夊瓧娈垫槧灏勬垨鎺ㄥ鍒?canonical schema銆?- 宸茬‘璁ょ己澶卞瓧娈典細琚ˉ鎴?`0.0`锛屽洜姝ゅ瓨鍦ㄢ€滈粯璁ゅ€兼帺鐩栫湡瀹炵己澶扁€濈殑椋庨櫓銆?
### 3.4 褰撳墠涓ゆ潯杩愯閾捐矾骞朵笉绛変环

#### 鐩稿瀹夊叏閾捐矾

- 鏂囦欢锛歚/home/jack/UAV-Fault-Diagnosis/apps/dashboard/backend/app/services/telemetry_manager.py`
- 宸茬‘璁?`_on_backend_frame()` 涓細
  - `backend_name == "px4"` 鏃朵細鍏堟墽琛?`adapt_px4_live_frame(frame)`锛?  - `backend_name == "replay"` 鏃朵細鍏堟墽琛?`canonicalize_rfly_frame(frame)`锛?  - 鐒跺悗鎵嶆妸 canonical frame / detect window 閫佸叆 detector銆?- 缁撹锛?*backend manager 閾捐矾涓庤缁冨悎鍚屾洿涓€鑷淬€?*

#### 楂橀闄╅摼璺?
- 鏂囦欢锛歚/home/jack/UAV-Fault-Diagnosis/src/online/realtime_detection_service.py`
- 宸茬‘璁?`_on_frame()` 涓洿鎺ユ墽琛?`self.detector.predict(list(window))`銆?- 宸茬‘璁よ澶勬病鏈夋樉寮忚皟鐢?`adapt_px4_live_frame()`銆?- 缁撹锛?*standalone realtime service 褰撳墠瀛樺湪 live schema 鐩存帴鍠傜粰 detector 鐨勯珮椋庨櫓銆?*

### 3.5 鏃堕棿鍩哄噯椋庨櫓

- 鏂囦欢锛?  - `/home/jack/UAV-Fault-Diagnosis/apps/dashboard/backend/app/services/detection_runner.py`
  - `/home/jack/UAV-Fault-Diagnosis/src/online/realtime_detection_service.py`
- 宸茬‘璁ゅ湪缂哄皯 `time_s` 鏃朵細閫€鍥?`frame_idx / 50.0`銆?- 缁撹锛氳嫢 PX4 live 瀹為檯閲囨牱鐜囧苟涓嶇ǔ瀹氬湪 50Hz锛屼簨浠舵椂闀垮拰 detection latency 瑙ｉ噴浼氫骇鐢熷亸宸€?
### 3.6 鏈鍚屾鏂板宸茬‘璁ら噷绋嬬

- 宸插舰鎴愭寮忔眹鎶ョ骇缁撹锛氬綋鍓嶅吋瀹规€у垽鏂淮鎸佷负 **Yellow / 鍙儴鍒嗚縼绉?*锛屼絾璇ョ粨璁轰粎鍦ㄢ€滆蛋 backend canonical 涓婚摼璺€濈殑鍓嶆彁涓嬫垚绔嬨€?- 宸茬‘璁ゅ綋鍓?MVP 鎺ㄨ崘鍩虹嚎搴旀敹鏁涗负锛歚PX4 live -> telemetry_manager -> adapt_px4_live_frame -> FaultDetector -> EventAggregator -> Dashboard/鏃ュ織`銆?- 宸茬‘璁?`src/online/realtime_detection_service.py` **涓嶅簲浣滀负褰撳墠 PX4 MVP 鍩虹嚎**锛屽洜涓鸿閾捐矾鐩存帴鎶?live schema 绐楀彛閫佸叆 detector銆?- 宸茬‘璁ゅ綋鍓嶆渶灏忓疄楠岃鍒掑簲鎸夆€滀簲姝モ€濇帹杩涳細杈撳叆鍏煎鎬ч棬绂?-> 闆舵牱鏈甯搁琛?-> 闆舵牱鏈紓甯搁琛?-> 鍏煎瀛愮壒寰侀泦 -> 瑙勫垯铻嶅悎銆?- 宸茬‘璁ゅ綋鍓嶉樆濉為」搴斿尯鍒嗕负涓ょ被锛?  1. 宸茬‘璁ら樆濉烇細閾捐矾 schema 涓嶄竴鑷淬€佹帶鍒?鎵ц鍣ㄨ涔夋湭瀹屽叏纭銆佺己灏戝彲閲嶅寮傚父鏍锋湰锛?  2. 寰呭疄娴嬮樆濉烇細[unverified] live 閲囨牱绋冲畾鎬с€佹湁鏁堢獥鍙ｆ瘮渚嬨€?-fill / sparse forward fill 瀵硅緭鍏ヨ川閲忕殑鎺╃洊绋嬪害銆佺湡瀹炲垎甯冨亸绉诲己搴︺€?
### 3.7 PX4 鎺у埗/鎵ц鍣ㄩ珮椋庨櫓瀛楁鐨勪唬鐮佺骇琛ュ厖纭

- 宸茬‘璁?`motor1..4` 鐨勯」鐩噰闆嗗叆鍙ｅ湪 `/home/jack/UAV-Fault-Diagnosis/src/telemetry/subscriptions.py`锛屾潵鑷?`actuator_output_status()`锛屽苟鍦ㄤ唬鐮佷腑鎸?`/1000.0` 褰掍竴鍖栧悗鍐欏叆 live schema銆?- 宸茬‘璁?`actuator_ctrl_*` 鐨勯」鐩噰闆嗗叆鍙ｅ湪 `/home/jack/UAV-Fault-Diagnosis/src/telemetry/pymav_supplement.py`锛屾潵鑷?`ACTUATOR_CONTROL_TARGET.controls[0:4]`銆?- 宸茬‘璁?`rates_sp_roll/pitch/yaw` 鐨勯」鐩噰闆嗗叆鍙ｅ湪 `/home/jack/UAV-Fault-Diagnosis/src/telemetry/pymav_supplement.py`锛屾潵鑷?`ATTITUDE_TARGET.body_*_rate`锛涘悓鏃?`/home/jack/PX4-Autopilot/msg/versioned/VehicleRatesSetpoint.msg` 宸茬‘璁?body rate setpoint 鍗曚綅涓?`rad/s`锛屽洜姝よ繖缁勫瓧娈靛湪鐗╃悊閲忕骇鍒笂涓?rate setpoint 璇箟鍩烘湰涓€鑷淬€?- 宸茬‘璁?`thrust_z` 鐨勯」鐩噰闆嗗叆鍙ｅ湪 `/home/jack/UAV-Fault-Diagnosis/src/telemetry/pymav_supplement.py`锛屽綋鍓嶇洿鎺ュ彇 `ATTITUDE_TARGET.thrust`锛涗絾 `/home/jack/PX4-Autopilot/msg/versioned/VehicleAttitudeSetpoint.msg` 宸叉彁绀?multicopter 涓嬪父瑙佺殑鏄?`thrust_body[2]` 涓旈€氬父甯﹁礋鍙风害瀹氾紝鍥犳 `thrust_z` 鐨勬柟鍚?绗﹀彿/閲忕翰涓€鑷存€т粛鏄珮椋庨櫓椤广€?- 宸茬‘璁?`/home/jack/PX4-Autopilot/msg/versioned/ActuatorMotors.msg` 涓?`control` 琛ㄧず normalized thrust `[-1,1]`锛涘悓鏃?`/home/jack/PX4-Autopilot/msg/ActuatorOutputs.msg` 琛ㄦ槑 `output` 榛樿鏄?natural output units锛屼粎鐗瑰畾 sim 杈撳嚭璺緞鎵嶆槑纭帴杩?`[-1,1]`銆傚洜姝ゅ綋鍓?`motor1..4 -> actuator_out_*` 鐨勨€滈噺绾插凡瀵归綈璁粌鍩熲€濊繖涓€鐐?*浠嶄笉鑳界洿鎺ヤ笅缁撹**銆?
## 4. 褰撳墠涓嶅簲杩囧害涓嬬粨璁虹殑閮ㄥ垎

浠ヤ笅鍐呭**灏氭湭瀹炴祴**锛屽彧鑳借涓?`[unverified]`锛?
- [unverified] PX4 live 鐨?`dt_median`銆乣dt_p95`銆乣valid_window_ratio`
- [unverified] 鐪熷疄 PX4 鏍锋湰涓?`mapped / derived / default` 姣斾緥
- [unverified] `actuator_out_*`銆乣actuator_ctrl_*`銆乣rates_sp_*`銆乣thrust_z` 鐨勮法骞冲彴璇箟涓€鑷存€?  - 褰撳墠鍙繘涓€姝ョ粏鍒嗕负锛?    - `rates_sp_*`锛氬凡纭閲囬泦鏉ユ簮涓?PX4 rate setpoint 鐗╃悊閲忕骇鍒熀鏈竴鑷达紝浣嗕笌璁粌鍩熸渶缁堝悓涔夋€т粛寰呭疄娴嬶紱
    - `thrust_z`锛氬綋鍓嶄粛灞炴渶楂橀闄╁瓧娈典箣涓€锛屼富瑕侀闄╁湪绗﹀彿銆佹柟鍚戜笌鎺ㄥ姏琛ㄨ揪鍘嬬缉锛?    - `actuator_out_*`锛氬凡纭 live 鏉ユ簮锛屼絾鏄惁涓庤缁冨煙 normalized actuator/motor 杈撳嚭鍚岄噺绾蹭粛寰呭疄娴嬶紱
    - `actuator_ctrl_*`锛氬凡纭鏉ヨ嚜 control target锛屼絾涓庤缁冨煙鎺у埗灞傝涔夋槸鍚﹀畬鍏ㄤ竴鑷翠粛寰呭疄娴嬨€?- [unverified] 榛樿鍊兼槸鍚﹀湪瀹為檯鍦ㄧ嚎绐楀彛涓崰鎹富瀵?- [unverified] 鐜版湁妯″瀷鍦?PX4 姝ｅ父椋炶涓婄殑璇姤姘村钩
- [unverified] 鐜版湁妯″瀷鍦?PX4 寮傚父宸ュ喌涓嬬殑姒傜巼鎶崌鑳藉姏

## 5. 褰撳墠闃舵寤鸿璺嚎

### 5.1 MVP 鍩虹嚎

寤鸿褰撳墠 MVP 鍙噰鐢ㄤ互涓嬮摼璺細

`PX4 live -> telemetry_manager -> adapt_px4_live_frame -> FaultDetector -> EventAggregator -> Dashboard/鏃ュ織`

涓嶅缓璁綋鍓嶆妸浠ヤ笅閾捐矾浣滀负杩佺Щ缁撹渚濇嵁锛?
`TelemetryReceiver -> RealtimeDetectionService -> FaultDetector`

### 5.2 鐩爣鑱氱劍

褰撳墠闃舵寤鸿缁х画鑱氱劍锛?
1. 杈撳叆鍏煎鎬ч獙璇侊紱
2. 鍦ㄧ嚎浜屽垎绫诲紓甯告娴嬮棴鐜紱
3. 鍙瀵熴€佸彲澶嶇幇銆佸彲鍥炴斁锛?4. 蹇呰鏃跺啀鍋氳鍒欒緟鍔┿€?
褰撳墠**涓嶅缓璁?*鐩存帴鎶婁富瑕佺洰鏍囨斁鍦細

- 杩佺Щ鍒?RflySim 骞冲彴鏈韩锛?- 閲嶆柊璁捐澶嶆潅妯″瀷锛?- 鐩存帴杩芥眰楂樼簿搴﹀鍒嗙被璇婃柇銆?
## 6. 鏈€灏忓疄楠屾墽琛岄『搴?
### Phase A锛氳緭鍏ュ吋瀹规€ч棬绂?
鍏堟敹闆嗕竴娈?PX4 姝ｅ父椋炶鏍锋湰锛屽彧鍋氳緭鍏ュ眰妫€鏌ワ細

- `mapped / derived / default` 姣斾緥
- 鍏抽敭瀛楁缂哄け鐜?/ 甯告暟鐜?- `dt_median`
- `dt_p95`
- `valid_window_ratio`
- 楂橀闄╁瓧娈佃〃

### Phase B锛氶浂鏍锋湰姝ｅ父椋炶璇曡窇

浠呭湪 Phase A 鍩烘湰閫氳繃鍚庯紝鍐嶅仛 backend canonical 閾捐矾涓嬬殑姝ｅ父椋炶闆舵牱鏈瘯璺戯紝鍏虫敞锛?
- `fault_prob` 鏄惁鎸佺画杈撳嚭锛?- 姝ｅ父娈垫槸鍚﹂暱鏈熸亽楂?/ 鎭掍綆锛?- event aggregation 鏄惁绋冲畾锛?- 鏄惁鍑虹幇鐖嗙偢鎬ц鎶ャ€?
### Phase C锛氶浂鏍锋湰寮傚父椋炶璇曡窇

鍦ㄥ彲鎺с€佸彲閲嶅寮傚父涓嬬户缁瀵燂細

- 寮傚父娈靛钩鍧囨鐜囨槸鍚︽槑鏄炬姮鍗囷紱
- 浜嬩欢鏁伴噺涓庢寔缁椂闂存槸鍚﹀彲瑙ｉ噴锛?- 鏄惁鍑虹幇棰戠箒闇囪崱鎴栧叏绋嬪紓甯稿父鏁伴娴嬨€?
### Phase D锛氳嫢涓嶇ǔ鍐嶅仛鏀剁缉

鑻ュ叏閲忕壒寰佷笉绋筹紝鍐嶈繘鍏ワ細

- 鍏煎瀛愮壒寰侀泦瀹為獙锛?- 瑙勫垯杈呭姪灞傚疄楠屻€?
## 7. 褰撳墠闃诲椤?
### 7.1 宸茬‘璁ら樆濉為」

1. `src/online/realtime_detection_service.py` 涓?backend canonical 涓婚摼璺笉涓€鑷达紝褰撳墠瀛樺湪 schema 椋庨櫓銆?2. `actuator_out_*`銆乣actuator_ctrl_*`銆乣rates_sp_*`銆乣thrust_z` 浠嶉渶鍋氭渶缁堣涔夌‘璁ゃ€?3. 褰撳墠杩樼己涓€濂楃敤浜庣ǔ瀹氬鐜扮殑 PX4 寮傚父鏍锋湰鎴栧紓甯告敞鍏ュ熀绾裤€?
### 7.2 寰呭疄娴嬮樆濉為」

- [unverified] PX4 live 瀹為檯閲囨牱鐜囨槸鍚︾ǔ瀹氬湪璁粌鍋囪闄勮繎銆?- [unverified] live 鏁版嵁鐨勬湁鏁堢獥鍙ｆ瘮渚嬫槸鍚﹁冻浠ユ敮鎾戣繛缁娴嬨€?- [unverified] 杩愯鏃?0-fill 涓?sparse forward fill 鏄惁鎺╃洊浜嗚緭鍏ヨ川閲忛棶棰樸€?- [unverified] RflySim / RflyMAD 鍒?PX4 鐨勫疄闄呭垎甯冨亸绉绘槸鍚﹀凡瓒充互褰卞搷闆舵牱鏈ǔ瀹氭€с€?
## 8. 褰撳墠鏈€鍙兘鐨勯棶棰樺眰绾?
- [x] 瀛楁鏄犲皠闂
- [x] 璇箟涓嶄竴鑷撮棶棰?- [x] 閲囨牱鐜?/ 鏃跺簭闂
- [ ] 缁熻鍒嗗竷鍋忕Щ闂锛圼unverified]锛?- [ ] 妯″瀷娉涘寲鑳藉姏闂锛圼unverified]锛?- [ ] 瑙勫垯灞傜己澶遍棶棰橈紙褰撳墠涓嶆槸绗竴浼樺厛绾э級

## 9. Phase A 杈撳叆鍏煎鎬ч棬绂佺殑鏈€灏忚惤鍦板彛寰勶紙浠ｇ爜绾у凡纭锛?
### 9.1 鍙洿鎺ュ鐢ㄧ殑鐜版湁瀹炵幇

- 宸茬‘璁ゅ彲鐩存帴澶嶇敤 `/home/jack/UAV-Fault-Diagnosis/src/telemetry/schema_adapter.py`锛?  - `normalize_time_columns()` 鍙粺涓€ `trueTime` / `Timestamp` 鍒?`time_s`锛?  - `adapt_px4_live_frame()` 宸茶兘杈撳嚭 canonical frame + `mapped / derived / default` coverage锛?  - `_apply_derived()` 浼氳ˉ榻?`31` 涓?canonical 杈撳叆鍒楋紝骞舵妸缂哄け鍩虹瀛楁钀藉埌 `default`銆?- 宸茬‘璁ゅ彲鐩存帴鍙傝€?`/home/jack/UAV-Fault-Diagnosis/scripts/eval_telemetry_live.py` 涓殑鏃堕棿缁熻鍙ｅ緞锛?  - `_infer_hz_from_time_s()` 璇佹槑椤圭洰鍐呭凡鏈夊熀浜?`time_s` 鐨?`dt` / 棰戠巼浼拌璺緞锛?  - `_time_array()` 鍙綔涓哄悗缁粺涓€鏃堕棿杞村鐞嗗弬鑰冦€?- 宸茬‘璁?`/home/jack/UAV-Fault-Diagnosis/src/telemetry/replay_metrics.py` 宸叉彁渚涗簨浠剁骇鍙ｅ緞锛涘洜姝?**Phase A 搴旇仛鐒﹁緭鍏ュ眰锛屼笉搴旀彁鍓嶆贩鍏?hit/miss/latency 缁撹**銆?
### 9.2 Phase A 鏈€灏戝簲杈撳嚭鐨勬寚鏍?
寤鸿瀵瑰崟娈?PX4 live 鏍锋湰鑷冲皯杈撳嚭浠ヤ笅 5 绫绘寚鏍囷細

1. `mapped_ratio / derived_ratio / default_ratio`锛氬熀浜庨€愬抚 `adapt_px4_live_frame()` coverage 鑱氬悎锛?2. `dt_median / dt_p95`锛氬熀浜?`time_s` 鐩搁偦宸垎锛?3. `valid_window_ratio`锛氭寜 `window_size=150` 缁熻锛岀獥鍙ｅ唴 `default` 鍗犳瘮涓嶈繃楂樹笖 `time_s` 鏃犻暱闂存柇鐨勭獥鍙ｆ瘮渚嬶紱
4. 鍏抽敭瀛楁 `missing_rate / constant_rate`锛氳嚦灏戣鐩?`actuator_out_*`銆乣actuator_ctrl_*`銆乣gyro_*`銆乣accel_*`銆乣vel_*`銆乣baro_*`銆乣att_q_*`銆乣mag_*`锛?5. 楂橀闄╁瓧娈靛崟鍒楁憳瑕侊細鑷冲皯鍗曠嫭鍒楀嚭 `actuator_out_*`銆乣actuator_ctrl_*`銆乣rates_sp_*`銆乣thrust_z`銆?
### 9.3 Phase A 寤鸿浜х墿

寤鸿鏈€灏忎骇鍑烘敹鏁涗负 3 浠芥枃浠讹細

- `phase_a_summary.json`锛氭€绘寚鏍囨憳瑕侊紝濡?`rows`銆乣time_range_s`銆乣dt_median`銆乣dt_p95`銆乣estimated_hz`銆乣valid_window_ratio`锛?- `phase_a_feature_coverage.csv`锛氶€?canonical 瀛楁鐨?`mapped/derived/default` 缁熻涓庡崰姣旓紱
- `phase_a_risk_fields.csv`锛氶珮椋庨櫓瀛楁鐨?`missing_rate`銆乣constant_rate`銆乣min/max/mean/std` 鎽樿銆?
### 9.4 褰撳墠寤鸿鎵ц椤哄簭

1. 鍏堝鍘熷 PX4 live CSV 鍋?`normalize_time_columns()`锛?2. 鍐嶉€愬抚鎵ц `adapt_px4_live_frame()` 骞剁疮璁?coverage锛?3. 鍐嶆寜 `window_size=150` 璁＄畻 `valid_window_ratio`锛?4. 鏈€鍚庡崟鐙緭鍑洪珮椋庨櫓瀛楁鎽樿锛屼笉鍦?Phase A 鎻愬墠鍋氭ā鍨嬩紭鍔ｇ粨璁恒€?
## 10. 涓嬩竴姝ヨ鍔?
1. 鐢ㄧ湡瀹?PX4 live 鏍锋湰杩愯 `scripts/phase_a_input_gate.py`锛岀敓鎴愰鎵?Phase A 缁熻浜х墿銆?2. 鍩轰簬 `phase_a_summary.json / phase_a_feature_coverage.csv / phase_a_risk_fields.csv` 鍋氶棬绂佸垽鏂€?3. 浠呭湪 Phase A 鍩烘湰閫氳繃鍚庯紝鎵嶈繘鍏?backend canonical 涓婚摼璺笅鐨勬甯搁琛岄浂鏍锋湰璇曡窇銆?4. 鍑嗗鍙噸澶嶅紓甯告牱鏈悗锛屽啀鍋氬紓甯搁琛岄浂鏍锋湰璇曡窇銆?5. 鑻ョ粨鏋滀笉绋筹紝鍐嶈浆鍏ュ吋瀹瑰瓙鐗瑰緛闆嗕笌瑙勫垯杈呭姪銆?6. 鎸佺画鎶婃柊杩涘睍鍚屾鍒版湰鏂囦欢銆?
## 11. 宸插悓姝ユ枃妗?
- 姝ｅ紡姹囨姤绋匡細`/home/jack/UAV-Fault-Diagnosis/docs/20260322_mvp_experiment_plan_report.md`
- MVP 姹囨姤绋匡細`/home/jack/UAV-Fault-Diagnosis/docs/20260322_px4_online_detection_mvp_report.md`
- 鏈枃浠剁敤浜庢寔缁洿鏂帮紱闃舵鎬ф眹鎶ョ鍙崟鐙敓鎴愶紝浣嗗凡楠岃瘉缁撹浠ユ湰鏂囦欢涓烘寔缁熀绾裤€?
## 12. Phase A 鍙墽琛岃剼鏈?/ 鍛戒护鍙ｅ緞锛堜笅涓€姝ヨ惤鍦板缓璁級

### 12.1 鐩爣

褰撳墠涓嬩竴姝ヤ笉搴旂洿鎺ヨ繘鍏ラ浂鏍锋湰妯″瀷璇曡窇锛岃€屽簲鍏堟妸 `Phase A锛氳緭鍏ュ吋瀹规€ч棬绂乣 鍙樻垚**鍙鐢ㄣ€佸彲钀界洏銆佸彲閲嶅鎵ц**鐨勭粺璁″叆鍙ｃ€?
鐩爣鏄鍗曟 PX4 live CSV 浜у嚭浠ヤ笅 3 浠藉浐瀹氭枃浠讹細

- `phase_a_summary.json`
- `phase_a_feature_coverage.csv`
- `phase_a_risk_fields.csv`

### 12.2 寤鸿瀹炵幇褰㈠紡

寤鸿浼樺厛鏂板涓€涓嫭绔嬭剼鏈紝鑱岃矗鍙仛鐒﹁緭鍏ュ眰锛屼笉娣峰叆妯″瀷鎺ㄧ悊缁撹銆?
寤鸿鑴氭湰鍚嶏細

- `scripts/phase_a_input_gate.py`

寤鸿杈撳叆鍙傛暟鏈€灏戝寘鎷細

- `--csv`
- `--output-dir`
- `--window-size`锛岄粯璁?`150`
- `--max-default-ratio`锛岀敤浜庡畾涔夋湁鏁堢獥鍙ｉ棬闄?- `--max-gap-s`锛岀敤浜庡畾涔夋椂闂撮暱闂存柇闂ㄩ檺

寤鸿鑴氭湰鍙仛 4 姝ワ細

1. 璇诲彇鍘熷 PX4 CSV锛屽苟閫氳繃 `normalize_time_columns()` 缁熶竴鍒?`time_s`锛?2. 閫愯鎵ц `adapt_px4_live_frame()`锛岀疮璁?`mapped / derived / default` coverage锛?3. 缁熻 `dt_median / dt_p95 / estimated_hz / valid_window_ratio`锛?4. 鍗曠嫭杈撳嚭楂橀闄╁瓧娈垫憳瑕侊紝涓嶅湪姝ら樁娈垫贩鍏?detector 缁撴灉銆?
### 12.3 寤鸿澶嶇敤浠ｇ爜鐐?
瀹炵幇鏃朵紭鍏堝鐢ㄤ互涓嬪凡鏈夐€昏緫锛?
- `src/telemetry/schema_adapter.py`
  - `normalize_time_columns()`
  - `adapt_px4_live_frame()`
- `scripts/eval_telemetry_live.py`
  - `_infer_hz_from_time_s()`
  - `_time_array()`

### 12.4 寤鸿鐨勬渶灏忓懡浠ゅ彛寰?
寤鸿鏈€缁堝澶栨敹鏁涙垚濡備笅鍛戒护褰㈠紡锛?
```bash
python scripts/phase_a_input_gate.py \
  --csv path/to/px4_live.csv \
  --output-dir results/phase_a_run_001 \
  --window-size 150 \
  --max-default-ratio 0.30 \
  --max-gap-s 0.20
```

### 12.5 寤鸿杈撳嚭瀛楁

#### `phase_a_summary.json`

寤鸿鑷冲皯鍖呭惈锛?
- `rows`
- `time_range_s`
- `dt_median`
- `dt_p95`
- `estimated_hz`
- `valid_window_ratio`
- `mapped_ratio`
- `derived_ratio`
- `default_ratio`
- `window_size`
- `max_default_ratio`
- `max_gap_s`

#### `phase_a_feature_coverage.csv`

寤鸿姣忚涓€涓?canonical 瀛楁锛岃嚦灏戝寘鍚細

- `feature`
- `mapped_count`
- `derived_count`
- `default_count`
- `mapped_ratio`
- `derived_ratio`
- `default_ratio`

#### `phase_a_risk_fields.csv`

寤鸿鑷冲皯瑕嗙洊浠ヤ笅瀛楁锛?
- `actuator_out_1..4`
- `actuator_ctrl_1..4`
- `rates_sp_roll/pitch/yaw`
- `thrust_z`

寤鸿鍒楀寘鎷細

- `feature`
- `missing_rate`
- `constant_rate`
- `min`
- `max`
- `mean`
- `std`
- `risk_note`

### 12.6 Phase A 閫氳繃 / 棰勮 / 澶辫触鍙ｅ緞锛堝缓璁級

褰撳墠寤鸿鍏堝湪鏂囨。涓浐鍖栧涓嬮棬绂佸彛寰勶紝鍚庣画瀹炴祴鍙啀璋冩暣锛?
- `閫氳繃`锛?  - `valid_window_ratio` 杈冮珮锛?  - 鍏抽敭 IMU / velocity / baro 瀛楁涓嶄互 `default` 涓轰富锛?  - 楂橀闄╁瓧娈佃櫧鏈夊亸绉伙紝浣嗘湭鍑虹幇澶ч潰绉己澶辨垨鎭掑畾鍊硷紱
- `棰勮`锛?  - 鍙舰鎴愮獥鍙ｏ紝浣嗛珮椋庨櫓瀛楁瀛樺湪杈冮珮 `default` 鎴?`constant_rate`锛?  - 鍙户缁繘鍏ラ浂鏍锋湰璇曡窇锛屼絾闇€鍚屾鍑嗗鍏煎瀛愮壒寰侀泦锛?- `澶辫触`锛?  - 鏈夋晥绐楀彛姣斾緥鏄庢樉涓嶈冻锛?  - 澶氫釜鍏抽敭瀛楁浠ラ粯璁ゅ€间富瀵硷紱
  - 鏃堕棿杞村瓨鍦ㄦ槑鏄鹃暱闂存柇锛?  - 涓嶅缓璁洿鎺ヨ繘鍏ユā鍨嬭瘯璺戙€?
### 12.7 鏈樁娈电粨鏉熷悗鐨勫敮涓€鎺ㄨ崘涓嬩竴姝?
Phase A 鑴氭湰鍙ｅ緞钀藉湴鍚庯紝涓嬩竴姝ユ墠寤鸿杩涘叆锛?
- backend canonical 涓婚摼璺笅鐨勬甯搁琛岄浂鏍锋湰璇曡窇銆?
鍦ㄦ涔嬪墠锛屼笉寤鸿鎶?standalone `RealtimeDetectionService` 鐨勭粨鏋滀綔涓?PX4 MVP 鍒ゆ柇渚濇嵁銆?
## 13. Phase A 鑴氭湰鍖栦笌娴嬭瘯灏佸彛鐘舵€侊紙鏈鍚屾锛?
### 13.1 宸插畬鎴愪簨椤?
- 宸叉柊澧?Phase A 鐙珛缁熻鍏ュ彛锛歚/home/jack/UAV-Fault-Diagnosis/scripts/phase_a_input_gate.py`
- 宸茬‘璁よ鑴氭湰褰撳墠鍙洿鎺ヨ緭鍑?3 浠藉浐瀹氫骇鐗╋細
  - `phase_a_summary.json`
  - `phase_a_feature_coverage.csv`
  - `phase_a_risk_fields.csv`
- 宸茶ˉ榻愬苟閫氳繃鐩稿叧娴嬭瘯锛?  - `tests/test_phase_a_input_gate.py`
  - `tests/test_eval_telemetry_live.py`
  - `tests/test_schema_adapter.py`

### 13.2 鏈宸插皝鍙ｇ殑鍏抽敭鐐?
- 宸插皝鍙?`Phase A` 鐨勬椂闂磋酱闂ㄧ锛?  - 鍧?`time_s` 涓嶅啀闈欓粯鍥為€€鎴?`arange`
  - 褰撳墠浼氭樉寮忚緭鍑?`time_axis_valid=0`
  - 鍦ㄦ椂闂磋酱鏃犳晥鏃讹紝`valid_window_ratio=0.0`
- 宸插皝鍙?broader key fields 绋冲畾鎬х粺璁★細
  - `phase_a_risk_fields.csv` 涓嶅啀鍙鐩?actuator/control 楂橀闄╁瓧娈?  - 褰撳墠宸茶鐩?`gyro_*`銆乣accel_*`銆乣vel_*`銆乣baro_*`銆乣att_q_*`銆乣mag_*`銆乣rates_sp_*`銆乣thrust_z`
- 宸插皝鍙?`eval_telemetry_live.py` 鐨?backend 鍒嗘祦鍙ｅ緞锛?  - 褰撳墠宸叉敮鎸佹樉寮?`--backend {replay,px4}`
  - `px4` 妯″紡涓嬭蛋 `adapt_px4_live_frame()`
  - `replay` 妯″紡涓嬩繚鎸?`canonicalize_rfly_records()`
  - 璇ュ垎娴佸凡琚?`main()` 绾ф祴璇曢攣瀹氾紝涓嶅啀鍙槸 helper 绾у亣璁?
### 13.3 褰撳墠鐘舵€佸垽鏂?
- 褰撳墠鍙垽瀹氾細**Phase A 宸插畬鎴愯剼鏈寲锛屼笖娴嬭瘯灏佸彛宸茶ˉ瀹屻€?*
- 杩欐剰鍛崇潃褰撳墠闃诲鐐瑰凡浠庘€滃彛寰勬湭钀藉湴 / 娴嬭瘯鏈皝鍙ｂ€濊浆绉诲埌鈥滅己灏戠湡瀹?PX4 live 鏍锋湰缁熻缁撴灉鈥濄€?- 鍥犳锛屼笅涓€姝ヤ笉搴旂户缁ˉ鏂规鏂囨。鎴栨娊璞¤璁★紝鑰屽簲杩涘叆锛?  - `鐪熷疄 PX4 live CSV -> Phase A 鑴氭湰 -> 棣栨壒闂ㄧ缁撴灉`

### 13.4 褰撳墠鍞竴鎺ㄨ崘涓嬩竴姝?
浣跨敤鐪熷疄 PX4 live 鏍锋湰鎵ц锛?
```bash
python scripts/phase_a_input_gate.py \
  --csv path/to/px4_live.csv \
  --output-dir results/phase_a_run_001 \
  --window-size 150 \
  --max-default-ratio 0.30 \
  --max-gap-s 0.20
```

鐒跺悗鍩轰簬浜х墿鍐冲畾鏄惁杩涘叆 `Phase B锛氶浂鏍锋湰姝ｅ父椋炶璇曡窇`銆?
## 14. 鏈钀界洏璇存槑

- 寤烘。鏃堕棿锛?026-03-22
- 鏈€杩戝悓姝ユ椂闂达細2026-03-22
- 璁板綍鏂瑰紡锛氬熀浜庝粨搴撴枃妗ｄ笌浠ｇ爜闃呰
- 缁撴灉鎬ц川锛?*褰撳墠宸蹭粠绾唬鐮佸璁￠樁娈垫帹杩涘埌鈥滆剼鏈凡钀藉湴 + 娴嬭瘯宸插皝鍙ｂ€濓紝浣嗕粛涓嶆槸瀹為鎴栧洖鏀惧疄楠岀粨鏋?*
