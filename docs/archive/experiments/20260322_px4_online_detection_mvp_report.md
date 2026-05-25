# PX4 鍦ㄧ嚎妫€娴?MVP 涓庢渶灏忓疄楠岃鍒掓眹鎶ョ

## 1. 缁撹鎽樿

褰撳墠缁撹寤鸿瀹氫负锛?*鍙儴鍒嗚縼绉伙紙Yellow锛?*锛屼笉寤鸿鐩存帴缁欏嚭鈥滃彲鐩存帴杩佺Щ鈥濈殑鍒ゆ柇銆?
鍘熷洜涓嶆槸鍦ㄧ嚎妫€娴嬪熀纭€璁炬柦缂哄け锛岃€屾槸**杈撳叆鍚堝悓鍦ㄤ笉鍚岃繍琛岃矾寰勪笂骞朵笉绛変环**锛?
- backend `TelemetryProcessManager` 璺緞浼氬厛鍋?canonical 鍖栵紝鍐嶈繘鍏?detector銆?- standalone `RealtimeDetectionService` 璺緞鐩存帴鎶?raw live window 閫佸叆 detector锛屽瓨鍦ㄨ緭鍏ュ悎鍚屾柇瑁傘€?
鍥犳锛岀幇闃舵鍙湁 **PX4 -> canonical -> detector** 杩欐潯 backend 璺緞鍏峰璇曡窇浠峰€硷紱standalone realtime path 鐩墠涓嶅簲浣滀负璺ㄥ钩鍙?MVP 涓昏矾寰勩€?
## 2. 浠ｇ爜绾т緷鎹?
### 2.1 detector 杈撳叆鍚堝悓寮虹粦瀹氳缁冨垪椤哄簭

`FaultDetector` 鐩存帴浠?`norm_stats.json` 璇诲彇 `columns`銆乣mean`銆乣std`锛屽苟鎸夎椤哄簭鏋勯€犳帹鐞嗚緭鍏ャ€?
- `/home/jack/UAV-Fault-Diagnosis/src/online/fault_detector.py:231`
- `/home/jack/UAV-Fault-Diagnosis/src/online/fault_detector.py:233`

鍚屾椂锛宒etector 瀵圭己澶卞瓧娈靛彧浼氳ˉ灏戦噺娲剧敓椤癸紱鍏朵綑瀛楁鏈€缁堜細閫€鍖栦负 `0.0`銆?
- `/home/jack/UAV-Fault-Diagnosis/src/online/fault_detector.py:273`
- `/home/jack/UAV-Fault-Diagnosis/src/online/fault_detector.py:336`

杩欐剰鍛崇潃 detector 涓嶈兘鏇夸唬瀹屾暣鐨?live -> canonical 鏄犲皠灞傘€?
### 2.2 standalone realtime path 浠嶅瓨鍦?schema 鏂

`RealtimeDetectionService._on_frame()` 鍦ㄧ獥鍙ｆ弧鍚庣洿鎺ユ墽琛岋細

- `/home/jack/UAV-Fault-Diagnosis/src/online/realtime_detection_service.py:62`
- `/home/jack/UAV-Fault-Diagnosis/src/online/realtime_detection_service.py:70`

杩欓噷浣跨敤鐨勬槸 `TelemetryReceiver` 鐨?raw live window锛岃€屼笉鏄?canonical window銆傚洜姝よ璺緞涓嬶紝澶ч噺 canonical 鍒楀彲鑳借惤鍏ラ粯璁ゅ€兼垨 `0.0`銆?
### 2.3 backend manager path 鏇存帴杩戝彲鐢?MVP

`TelemetryProcessManager._on_backend_frame()` 浼氬 PX4 甯у厛鎵ц `adapt_px4_live_frame()`锛屽苟璁板綍 `feature_coverage`銆?
- `/home/jack/UAV-Fault-Diagnosis/apps/dashboard/backend/app/services/telemetry_manager.py:789`
- `/home/jack/UAV-Fault-Diagnosis/apps/dashboard/backend/app/services/telemetry_manager.py:795`
- `/home/jack/UAV-Fault-Diagnosis/apps/dashboard/backend/app/services/telemetry_manager.py:803`

鎵€浠ュ綋鍓嶆洿鍙俊鐨勮瘯璺戝叆鍙ｅ簲鏄?backend manager path锛岃€屼笉鏄?standalone realtime service銆?
## 3. 鍏煎鎬у垽鏂?
### 3.1 褰撳墠鍒ゆ柇

寤鸿缁撹锛?*鍙儴鍒嗚縼绉?*銆?
### 3.2 鍘熷洜鎷嗚В

褰撳墠涓昏椋庨櫓涓嶅湪妯″瀷缁撴瀯锛岃€屽湪浠ヤ笅涓夊眰锛?
1. **瀛楁鏄犲皠闂**
   - live schema銆乧anonical schema銆佽缁?schema 骞朵笉澶╃劧涓€鑷淬€?2. **璇箟涓嶄竴鑷撮棶棰?*
   - `actuator_out_*`銆乣actuator_ctrl_*`銆乣rates_sp_*`銆乣thrust_z` 浠嶇己璺ㄥ钩鍙拌涔夌瓑浠烽獙璇併€?3. **閲囨牱鐜?/ 鏃跺簭闂**
   - `50Hz / 150` 绐楀彛鍋囪杩樻病鏈夌湡瀹?PX4 鏍锋湰瀹炴祴鏀拺銆?
褰撳墠**涓嶅缓璁紭鍏堟妸闂褰掑洜浜庢ā鍨嬫硾鍖栬兘鍔?*銆?
## 4. MVP 璺嚎寤鸿

寤鸿鎶婂綋鍓?MVP 涓昏矾寰勯攣瀹氫负锛?
**PX4 live -> schema adapter -> canonical window -> detector -> event aggregator -> coverage / logs / dashboard**

涓嶅缓璁妸 standalone `RealtimeDetectionService` 浣滀负鐜伴樁娈靛澶栧绉扮殑 MVP 鍏ュ彛銆?
### 鎺ㄨ崘鎺ㄨ繘椤哄簭

鎸夋枃妗ｅ畾涔夛紝浠嶉噰鐢?0/1/2/3/4 闃舵鎺ㄨ繘锛屼絾褰撳墠鍏堢‘璁ゆ渶灏忛棴鐜細

- **闃舵 0**锛氬疄鏃惰緭鍏ラ摼璺ǔ瀹?- **闃舵 1**锛歴chema 瑕嗙洊鐜囧拰 canonical 杈撳叆璐ㄩ噺杈炬爣
- **闃舵 2**锛氭寔缁緭鍑?`fault_prob`锛屼笖浜嬩欢鑱氬悎绋冲畾
- **闃舵 3**锛氳嫢璇姤鍋忛珮锛屽啀鍔犺交瑙勫垯灞?- **闃舵 4**锛氬缓绔嬪彲閲嶅 replay 鍩虹嚎

鐜伴樁娈电殑閲嶇偣涓嶆槸杩芥眰鏈€缁堢簿搴︼紝鑰屾槸鍏堝缓绔?*鍙繍琛屻€佸彲瑙傚療銆佸彲鍥炴斁**鐨勯棴鐜€?
## 5. 鏈€灏忓疄楠屾墽琛岃鍒?
### 瀹為獙 1锛氳緭鍏ュ吋瀹规€у疄楠?
鐩爣锛氬厛纭 PX4 鏁版嵁鏄惁婊¤冻 detector 杈撳叆鍚堝悓銆?
寤鸿鍔ㄤ綔锛?
- 浠呰蛋 backend manager canonical path銆?- 閲囬泦 PX4 姝ｅ父椋炶鏍锋湰銆?- 杈撳嚭浠ヤ笅鎸囨爣锛?  - `mapped / derived / default`
  - 鍏抽敭瀛楁缂哄け鐜?  - 鍏抽敭瀛楁甯告暟鐜?  - `dt_median`
  - `dt_p95`
  - `valid_window_ratio`

鍒ゅ畾鎰忎箟锛氳繖鏄?L1 / L3 / L5 鐨勫墠缃棬妲涳紝涓嶉€氳繃鏃朵笉搴旀€ヤ簬璁ㄨ妯″瀷鏁堟灉銆?
### 瀹為獙 2锛氶浂鏍锋湰杩佺Щ瀹為獙

鐩爣锛氶獙璇佺幇鏈?Rfly 璁粌妯″瀷鍦?PX4 canonical 杈撳叆涓嬫槸鍚﹁兘绋冲畾杈撳嚭銆?
寤鸿鍔ㄤ綔锛?
- 浠嶅彧璧?canonicalized backend path銆?- 鍏堢敤姝ｅ父椋炶鏁版嵁瑙傚療锛?  - `fault_prob` 鏄惁鎸佺画杈撳嚭
  - 鏄惁闀挎湡鎭掗珮 / 鎭掍綆
  - 浜嬩欢鏁版槸鍚﹀紓甯哥垎鐐?  - 浜嬩欢鑱氬悎鏄惁绋冲畾
- 鏈夊彲鎺у紓甯告暟鎹悗锛屽啀楠岃瘉寮傚父娈靛搷搴斻€?
### 瀹為獙 3锛氬吋瀹瑰瓙鐗瑰緛闆嗗疄楠?
浠呭湪鍏ㄩ噺鐗瑰緛涓嶇ǔ鏃惰繘鍏ャ€?
浼樺厛椤哄簭寤鸿锛?
1. `Core-IMU`
2. `Core-Flight`
3. `Core-Actuator`
4. `Full-Compatible`

鍘熷洜锛氬綋鍓嶉珮椋庨櫓闆嗕腑鍦?actuator/control 鐩稿叧瀛楁锛屽厛鐢ㄦ洿绋冲畾鐨勫熀纭€浼犳劅鍣ㄤ笌椋炶鐘舵€佸瓧娈垫洿鍚堢悊銆?
### 瀹為獙 4锛氳鍒欒瀺鍚堝疄楠?
浠呭湪妯″瀷宸叉湁涓€瀹氬搷搴斻€佷絾璇姤鎴栦笉绋冲畾杈冩槑鏄炬椂杩涘叆銆?
浼樺厛瑙勫垯锛?
- 鏁版嵁璐ㄩ噺瑙勫垯
- 绠€鍗曠墿鐞嗕竴鑷存€ц鍒?
鐜伴樁娈典笉寤鸿鎶娾€滃厛閲嶈妯″瀷鈥濅綔涓虹涓€閫夋嫨銆?
## 6. 褰撳墠闃诲椤?
1. `RealtimeDetectionService` 鏈?canonicalize锛屼笉鑳戒綔涓虹洿鎺ヨ縼绉荤粨璁轰緷鎹€?2. 缂哄皯鐪熷疄 PX4 鏍锋湰涓嬬殑 `feature_coverage` 瀹炴祴缁撴灉銆?3. 缂哄皯 `50Hz / 150` 绐楀彛閲囨牱绋冲畾鎬у疄娴嬫暟鎹€?4. `actuator_out_* / actuator_ctrl_* / rates_sp_* / thrust_z` 缂哄皯璺ㄥ钩鍙拌涔夌瓑浠烽獙璇併€?5. 缂哄皯 replay 鍩虹嚎鏍锋湰锛屾棤娉曡繘琛屽彲閲嶅鍥炴斁楠岃瘉銆?
## 7. 鍙澶栨眹鎶ュ彛寰?
寤鸿浣跨敤浠ヤ笅鍙ｅ緞锛?
- 鐜版湁 Rfly 璁粌鍦ㄧ嚎妫€娴嬪櫒**涓嶆槸瀹屽叏涓嶈兘杩佺Щ鍒?PX4**銆?- 浣嗗綋鍓嶅彧鑳界‘璁わ細
  - backend canonical 璺緞鍏峰璇曡窇浠峰€硷紱
  - standalone raw 璺緞鏆備笉鍏峰鐩存帴璇曡窇浠峰€笺€?- 鍥犳锛岄」鐩幇闃舵鏈€鍚堢悊璺嚎涓嶆槸绔嬪嵆姣旇緝妯″瀷绮惧害鎴栫洿鎺ラ噸璁紝鑰屾槸浼樺厛瀹屾垚锛?  1. backend canonical 杈撳叆璐ㄩ噺楠岃瘉
  2. 杩炵画杈撳嚭楠岃瘉
  3. 涓€鏉″彲鍥炴斁鍩虹嚎

## 8. 涓€鍙ヨ瘽缁撹

褰撳墠鏈€鍏抽敭鐨勪笉鏄户缁璁烘ā鍨嬩笂闄愶紝鑰屾槸鍏堥獙璇?**PX4 鏁版嵁鏄惁鍦?backend canonical 璺緞涓婄湡瀹炴弧瓒?detector 鐨勮緭鍏ュ悎鍚岋紝骞惰兘绋冲畾褰㈡垚鍦ㄧ嚎妫€娴嬮棴鐜?*銆?