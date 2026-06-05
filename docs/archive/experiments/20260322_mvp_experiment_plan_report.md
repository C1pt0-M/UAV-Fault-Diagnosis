# PX4 鍦ㄧ嚎妫€娴嬪吋瀹规€у垽鏂笌 MVP/瀹為獙璁″垝姹囨姤绋?
## 1. 缁撹鎽樿

鍩轰簬鐜版湁浠ｇ爜涓庢柟妗堟枃妗ｏ紝褰撳墠鍙粰鍑虹殑姝ｅ紡缁撹鏄細**PX4 -> canonical schema -> online detector 閾捐矾鍏峰鈥滃彲閮ㄥ垎杩佺Щ锛圷ellow锛夆€濇潯浠讹紝浣嗕笉寤鸿鐩存帴鎶婃墍鏈夌幇鏈夊疄鏃舵娴嬪叆鍙ｉ兘瑙嗕负鍚岀瓑鍙敤銆?*

鏇村叿浣撳湴璇达細

- 宸茬‘璁よ缁冧晶鍦ㄧ嚎妯″瀷渚濊禆鍥哄畾杈撳叆鍚堝悓锛歚31` 閫氶亾銆乣window_size=150`銆乣target_hz=50`锛屽苟鐢?`norm_stats.json.columns` 鍐冲畾鎺ㄧ悊鍙栧€奸『搴忋€?- 宸茬‘璁ら」鐩瓨鍦ㄥ彲鐢ㄧ殑 PX4 live -> canonical 閫傞厤灞傦紝鍙妸 PX4 瀛楁鏄犲皠鍒拌缁冩椂浣跨敤鐨?`SEQ_COLUMNS`銆?- 宸茬‘璁?`apps/dashboard/backend/app/services/telemetry_manager.py` 浼氬湪 PX4 璺緞涓婂厛 canonicalize 鍐嶉€?detector锛屽洜姝よ繖鏉￠摼璺笌璁粌鍚堝悓鏇翠竴鑷淬€?- 宸茬‘璁?`src/online/realtime_detection_service.py` 鐩存帴鎶?`TelemetryReceiver.window` 閫佸叆 `FaultDetector.predict()`锛屾病鏈夌湅鍒板悓绛夌殑 canonical 鏄犲皠姝ラ锛屽洜姝ゅ畠**涓嶉€傚悎浣滀负褰撳墠 PX4 MVP 鐨勫熀绾块摼璺?*銆?- [unverified] 鐪熷疄 PX4 live 鏁版嵁涓嬬殑 `dt_median`銆乣dt_p95`銆乣valid_window_ratio`銆乣mapped/default` 瀹炴祴姣斾緥浠嶆湭瀹屾垚锛屽洜姝ゅ綋鍓嶅垽鏂粛灞炰簬鈥滀唬鐮佷笌鏂规灞傞潰鐨勫彲閮ㄥ垎杩佺Щ鈥濓紝涓嶆槸鏈€缁堝疄楠岀粨璁恒€?
## 2. 宸茬‘璁や緷鎹?
### 2.1 璁粌涓庢帹鐞嗚緭鍏ュ悎鍚?
- `src/data/build_online_cache.py:27`-`30` 鍥哄畾浜嗗湪绾跨紦瀛樼殑鍏抽敭鍚堝悓锛歚WINDOW_SIZE=150`銆乣STRIDE=50`銆乣TARGET_HZ=50.0`銆?- `src/data/build_online_cache.py:25` 鏄剧ず鍦ㄧ嚎缂撳瓨澶嶇敤浜?`build_sequence_cache.py` 鐨?`SEQ_COLUMNS`銆?- `src/online/fault_detector.py:7`-`9` 涓?`src/online/fault_detector.py:14` 璇存槑鎺ㄧ悊杈撳叆閫氶亾椤哄簭鐢辨ā鍨嬬洰褰曚腑鐨?`norm_stats.json.columns` 鍐冲畾銆?
### 2.2 PX4 canonical 鍖栬兘鍔?
- `src/telemetry/schema_adapter.py:17`-`25` 宸插畾涔?`PX4_MAP`锛岃鐩?`actuator_out_*`銆乣actuator_ctrl_*`銆乣gyro_*`銆乣accel_*`銆乣baro_*`銆乣vel_*`銆乣mag_*`銆乣att_q_*` 绛夋牳蹇冨瓧娈垫槧灏勩€?- `src/telemetry/schema_adapter.py:66`-`90` 浼氳ˉ榻?`actuator_mean`銆乣actuator_std`銆乣acc_norm`銆乣actuator_effort`锛屽苟杈撳嚭 `mapped / derived / default` 瑕嗙洊淇℃伅銆?
### 2.3 褰撳墠鍙敤鐨勫畨鍏ㄤ富閾捐矾

- `apps/dashboard/backend/app/services/telemetry_manager.py:794`-`799` 鍦?`px4` / `replay` 璺緞涓婂厛鎵ц `adapt_px4_live_frame()` 鎴?`canonicalize_rfly_frame()`銆?- `apps/dashboard/backend/app/services/telemetry_manager.py:802`-`810` 浼氳褰?`feature_coverage`銆?- `apps/dashboard/backend/app/services/telemetry_manager.py:826` 鍦?canonical frame / canonical window 鍩虹涓婅皟鐢?detector銆?
### 2.4 褰撳墠涓嶅疁浣滀负 MVP 鍩虹嚎鐨勯摼璺?
- `src/online/realtime_detection_service.py:66`-`70` 鍦ㄧ獥鍙ｆ弧鍚庣洿鎺ユ墽琛?`self.detector.predict(list(window))`銆?- 璇?`window` 鏉ヨ嚜 `TelemetryReceiver` 鐨?live schema锛岃€屼笉鏄缁冧晶 canonical schema銆?- 鍥犳杩欐潯閾捐矾涓嬶紝canonical 鍒楃己澶变細鏇村鏄撻€€鍖栨垚 detector 鍐呴儴鐨勭己鐪佸彇鍊兼垨 0 鍊煎～鍏呫€?
## 3. 鍏煎鎬у垽鏂?
### 3.1 褰撳墠鍒ゆ柇

**缁撹锛歒ellow / 鍙儴鍒嗚縼绉汇€?*

鍘熷洜濡備笅锛?
1. 杈撳叆鍚堝悓骞堕潪瀹屽叏鏂锛歅X4 live 鍒?canonical 鐨勯€傞厤灞傚凡瀛樺湪锛屾牳蹇?IMU銆乿elocity銆乥aro銆乵ag銆乤ctuator 瀛楁鏈夋槑纭槧灏勬潵婧愩€?2. 涓婚摼璺苟闈炲畬鍏ㄧ己澶憋細鍚庣 `telemetry_manager` 宸插叿澶?canonicalize -> detect -> event aggregate 鐨勪富璺緞銆?3. 浣嗗綋鍓嶄笉鑳藉垽鏂负 Green锛?   - [unverified] 灏氭棤鐪熷疄 PX4 live 閲囨牱绋冲畾鎬х粺璁★紱
   - [unverified] 灏氭棤鐪熷疄 `default` 瀛楁鍗犳瘮鍜屾湁鏁堢獥鍙ｅ崰姣旓紱
   - `actuator_out_*`銆乣actuator_ctrl_*`銆乣rates_sp_*`銆乣thrust_z` 浠嶅瓨鍦ㄨ涔変笌閲忕翰纭闂锛?   - standalone realtime service 涓?backend manager 鐨勮涓虹洰鍓嶄笉绛変环銆?
### 3.2 褰撳墠鏈€鍙兘鐨勯棶棰樺眰绾?
- 绗竴灞傦細瀛楁鏄犲皠涓庨摼璺竴鑷存€ч棶棰?- 绗簩灞傦細鎺у埗/鎵ц鍣ㄧ浉鍏冲瓧娈电殑璇箟涓€鑷存€ч棶棰?- 绗笁灞傦細閲囨牱鐜囦笌绐楀彛绋冲畾鎬ч棶棰?- 绗洓灞傦細[unverified] 缁熻鍒嗗竷鍋忕Щ闂

## 4. MVP 璺嚎寤鸿

### 4.1 鎺ㄨ崘璺嚎

褰撳墠 MVP 搴斾紭鍏堟敹鏁涗负锛?*PX4 瀹炴椂閬ユ祴 -> `telemetry_manager` canonical 鍖?-> 鍦ㄧ嚎浜屽垎绫?detector -> event aggregation -> Dashboard/鏃ュ織瑙傛祴**銆?
### 4.2 杩欐牱鍋氱殑鍘熷洜

- 璇ヨ矾绾垮凡缁忓叿澶?schema 閫傞厤銆乧overage 鏆撮湶銆佷簨浠惰仛鍚堝拰鍚庣鐘舵€佽娴嬭兘鍔涖€?- 璇ヨ矾绾夸笌璁粌鍚堝悓鏇翠竴鑷达紝鏇撮€傚悎鍏堥獙璇佲€滆兘鍚︾ǔ瀹氬舰鎴愬疄鏃舵娴嬮棴鐜€濄€?- 杩欎篃绗﹀悎 `docs/realtime_detection_mvp_plan.md` 鐨勭洰鏍囷細鍏堝缓绔嬭繛缁緭鍏ャ€佺ǔ瀹氭粦绐椼€佽繛缁緭鍑哄拰鍙В閲婂憡璀︼紝鑰屼笉鏄洿鎺ヨ拷姹傛洿澶嶆潅鐨勫疄鏃惰瘖鏂郴缁熴€?
### 4.3 褰撳墠闃舵涓嶅缓璁仛鐨勪簨

- 涓嶅缓璁妸 `src/online/realtime_detection_service.py` 鐩存帴褰撲綔 PX4 MVP 鍩虹嚎銆?- 涓嶅缓璁湪鏈畬鎴愯緭鍏ュ吋瀹规€ч棬绂佸墠锛岀洿鎺ヤ互绮惧害鎴栧憡璀﹀懡涓巼璇勪环妯″瀷濂藉潖銆?- 涓嶅缓璁湪鎺у埗/鎵ц鍣ㄨ涔夋湭纭鍓嶏紝璐哥劧鎶婂叏閲忔帶鍒剁被鐗瑰緛瑙嗕负鈥滃彲淇¤縼绉荤壒寰佲€濄€?
## 5. 鏈€灏忓疄楠屾墽琛岃鍒?
寤鸿鎸変互涓嬮『搴忔帹杩涳紝鍓嶄竴姝ヤ笉閫氳繃鍒欎笉瑕佽繘鍏ュ悗涓€姝ャ€?
### 闃舵 A锛氳緭鍏ュ吋瀹规€ч棬绂?
鐩爣锛氬厛鍒ゆ柇 PX4 鏁版嵁鏄惁鑳芥瀯鎴愨€滃舰鐘舵纭€濈殑 detector 杈撳叆銆?
鏈€灏戣緭鍑猴細

- `mapped / derived / default` 姣斾緥
- 鍏抽敭瀛楁缂哄け鐜?/ 甯告暟鐜?- `dt_median`銆乣dt_p95`
- `valid_window_ratio`
- 楂橀闄╁瓧娈佃〃

闃舵鍒ゅ畾锛?
- 閫氳繃锛氳繘鍏ラ浂鏍锋湰杩佺Щ瀹為獙
- 璀﹀憡锛氳繘鍏ラ浂鏍锋湰瀹為獙锛屼絾鍚屾鍑嗗鍏煎瀛愮壒寰侀泦
- 澶辫触锛氬仠姝㈡ā鍨嬬粨璁猴紝鍏堝鐞嗚緭鍏ュ悎鍚岄棶棰?
### 闃舵 B锛氶浂鏍锋湰姝ｅ父椋炶璇曡窇

鐩爣锛氬彧鐢ㄦ甯?PX4 鏁版嵁锛岄獙璇?fault probability 鏄惁杩炵画銆佺ǔ瀹氥€佸彲瑙ｉ噴銆?
閲嶇偣瑙傚療锛?
- 鏄惁鎸佺画杈撳嚭 `fault_prob`
- 鏄惁鍑虹幇闀挎湡鎭掗珮 / 鎭掍綆
- 浜嬩欢鑱氬悎鏄惁绋冲畾
- 姝ｅ父娈垫槸鍚﹀嚭鐜扮垎鐐告€ц鎶?
### 闃舵 C锛氶浂鏍锋湰寮傚父椋炶璇曡窇

鐩爣锛氬湪鍙帶銆佸彲閲嶅寮傚父涓嬮獙璇佸紓甯告鏄惁瀛樺湪姒傜巼鎶崌涓庝簨浠跺搷搴斻€?
閲嶇偣瑙傚療锛?
- 寮傚父娈靛钩鍧囨鐜囨槸鍚﹂珮浜庢甯告
- 浜嬩欢鏁伴噺涓庢寔缁椂闂存槸鍚﹀彲瑙ｉ噴
- 鏄惁鍑虹幇棰戠箒闇囪崱鎴栧叏绋嬪紓甯稿父鏁伴娴?
### 闃舵 D锛氬吋瀹瑰瓙鐗瑰緛闆嗗疄楠?
浠呭綋鍏ㄩ噺鐗瑰緛闆舵牱鏈縼绉讳笉绋冲畾鏃惰繘鍏ャ€?
寤鸿浼樺厛姣旇緝锛?
1. Core-IMU
2. Core-Flight
3. Core-Actuator
4. Full-Compatible

鐩爣鏄洖绛旓細闂涓昏鏉ヨ嚜楂橀闄╁瓧娈碉紝杩樻槸鏉ヨ嚜鏇存繁灞傜殑 domain shift銆?
### 闃舵 E锛氳鍒欒瀺鍚堝疄楠?
浠呭綋妯″瀷杈撳嚭宸插彲杩愯浣嗙ǔ瀹氭€т粛涓嶈冻鏃惰繘鍏ャ€?
寤鸿浼樺厛寮曞叆锛?
- 鏁版嵁璐ㄩ噺瑙勫垯
- 鐗╃悊涓€鑷存€ц鍒?- 鐩稿叧鎬ц鍒?
鐩爣涓嶆槸鏇夸唬妯″瀷锛岃€屾槸澧炲己宸ョ▼绋冲畾鎬т笌鍙В閲婃€с€?
## 6. 褰撳墠闃诲椤?
### 6.1 宸茬‘璁ら樆濉為」

1. `src/online/realtime_detection_service.py` 涓?backend canonical 涓婚摼璺笉涓€鑷达紝褰撳墠瀛樺湪 schema 椋庨櫓銆?2. 鎺у埗/鎵ц鍣ㄧ浉鍏冲瓧娈典粛闇€鍋氭渶缁堣涔夌‘璁わ紝灏ゅ叾鏄?`actuator_out_*`銆乣actuator_ctrl_*`銆乣rates_sp_*`銆乣thrust_z`銆?3. 褰撳墠杩樼己涓€濂楃敤浜庣ǔ瀹氬鐜扮殑 PX4 寮傚父鏍锋湰鎴栧紓甯告敞鍏ュ熀绾裤€?
### 6.2 寰呭疄娴嬮樆濉為」

1. [unverified] PX4 live 瀹為檯閲囨牱鐜囨槸鍚︾ǔ瀹氬湪璁粌鍋囪闄勮繎銆?2. [unverified] live 鏁版嵁鐨勬湁鏁堢獥鍙ｆ瘮渚嬫槸鍚﹁冻浠ユ敮鎾戣繛缁娴嬨€?3. [unverified] 杩愯鏃?0-fill 涓?sparse forward fill 鏄惁鍦ㄧ湡瀹為摼璺腑鎺╃洊浜嗚緭鍏ヨ川閲忛棶棰樸€?4. [unverified] RflySim -> PX4 鐨勫疄闄呭垎甯冨亸绉绘槸鍚﹀凡杈惧埌褰卞搷闆舵牱鏈縼绉荤ǔ瀹氭€х殑绋嬪害銆?
## 7. 瀵瑰鍙洿鎺ュ紩鐢ㄧ殑姹囨姤缁撹

> 褰撳墠 PX4 鍦ㄧ嚎妫€娴嬮摼璺凡缁忓叿澶団€滃彲閮ㄥ垎杩佺Щ鈥濈殑鍩虹锛氳缁冭緭鍏ュ悎鍚屾槑纭€丳X4 鍒?canonical 鐨勯€傞厤灞傚瓨鍦ㄣ€乥ackend 涓昏矾寰勮兘澶熷湪 canonical 鍖栧悗鎵ц瀹炴椂妫€娴嬩笌浜嬩欢鑱氬悎銆備絾鐜伴樁娈垫洿閫傚悎鍏堝仛鈥滃疄鏃跺紓甯告娴?MVP鈥濓紝鑰屼笉鏄洿鎺ヨ拷姹傚畬鏁村疄鏃惰瘖鏂€傚叿浣撴墽琛屼笂锛屽簲浼樺厛浣跨敤 `telemetry_manager` 杩欐潯 canonical 涓婚摼璺紝鎸夆€滆緭鍏ュ吋瀹规€ч棬绂?-> 闆舵牱鏈甯搁琛?-> 闆舵牱鏈紓甯搁琛?-> 鍏煎瀛愮壒寰侀泦 -> 瑙勫垯铻嶅悎鈥濈殑椤哄簭鎺ㄨ繘锛屽苟浼樺厛琛ラ綈 live 閲囨牱绋冲畾鎬с€佺獥鍙ｆ湁鏁堢巼銆佹帶鍒惰涔夌‘璁ゅ拰鍙噸澶嶅紓甯告牱鏈繖鍥涚被闃诲椤广€?