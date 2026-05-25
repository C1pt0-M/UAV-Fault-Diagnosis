# 瀹炴椂閬ユ祴涓庢晠闅滄娴嬫ā鍧?
## 妯″潡璇存槑

`src/telemetry/` - 瀹炴椂閬ユ祴閲囬泦涓庝紶杈撴ā鍧?`src/online/realtime_detection_service.py` - 瀹炴椂鏁呴殰妫€娴嬫湇鍔?
## 鏋舵瀯

```
鏃犱汉鏈轰晶 (collector)  鈫? UDP 浼犺緭  鈫? 浜戠 (receiver + detector)
     鈫?                                       鈫?  MAVSDK/pymavlink                    瀹炴椂鏁呴殰妫€娴?+ 浜嬩欢鑱氬悎
  閲囬泦 PX4 閬ユ祴                        CNN1D 鎺ㄧ悊 + 鍛婅
  50Hz 蹇収                            CSV 璁板綍
```

## 浣跨敤鏂规硶

### 1. 鍚姩 PX4 SITL 浠跨湡

```bash
cd ~/PX4-Autopilot
make px4_sitl gazebo-classic
```

### 2. 鍚姩浜戠瀹炴椂妫€娴嬫湇鍔?
```bash
cd ~/UAV-Fault-Diagnosis
python -m online.realtime_detection_service --model assets/models/online_detector_CNN
```

鍙傛暟锛?- `--model` - 妯″瀷鐩綍锛堝繀闇€锛?- `--port` - UDP 鐩戝惉绔彛锛堥粯璁?15550锛?- `--csv` - CSV 杈撳嚭璺緞锛堥粯璁よ嚜鍔ㄧ敓鎴愶級
- `--window` - 婊戝姩绐楀彛澶у皬锛堥粯璁?150锛?- `--device` - 鎺ㄧ悊璁惧锛坈pu/cuda锛?
### 3. 鍚姩鏃犱汉鏈轰晶閬ユ祴閲囬泦

```bash
cd ~/Fault\ Dataset
python -m telemetry.collector
```

鍙傛暟锛?- `--host` - UDP 杞彂鐩爣鍦板潃锛堥粯璁?127.0.0.1锛?- `--port` - UDP 杞彂鐩爣绔彛锛堥粯璁?15550锛?- `--emit-rate` - 鍙戦€侀鐜?Hz锛堥粯璁?50锛?- `--no-pymav` - 绂佺敤 pymavlink 琛ュ厖

### 4. 鎺у埗鏃犱汉鏈洪琛?
浣跨敤 QGroundControl 鎴?MAVSDK 鑴氭湰鎺у埗鏃犱汉鏈鸿捣椋炪€侀琛屻€?
瀹炴椂妫€娴嬫湇鍔′細锛?- 鎺ユ敹閬ユ祴鏁版嵁骞惰褰曞埌 CSV
- 姣?150 甯э紙3 绉掞級杩愯涓€娆℃晠闅滄娴?- 妫€娴嬪埌鏁呴殰鏃剁珛鍗虫墦鍗板憡璀?- 椋炶缁撴潫鍚庤緭鍑哄畬鏁寸粺璁?
## 杈撳嚭绀轰緥

```
[realtime] 鏁呴殰妫€娴嬫湇鍔″凡鍒濆鍖?  妯″瀷: assets/models/online_detector_CNN
  绐楀彛: 150 甯?  绔彛: 15550
  CSV: telemetry_realtime_1773573000.csv
[receiver] 鐩戝惉 UDP :15550
[receiver] 璁板綍鍒?telemetry_realtime_1773573000.csv
[FaultDetector] 妯″瀷鍔犺浇瀹屾垚: cnn1d | 106,114 鍙傛暟, device=cpu
[realtime] 寮€濮嬫帴鏀堕仴娴嬫暟鎹苟瀹炴椂妫€娴?...
[receiver] 宸叉帴鏀?500 甯?| 涓㈠寘 0 (0.0%) | 鍧忓寘 0
[realtime] 宸叉娴?100 娆?| 褰撳墠 fault_prob=0.023 | 浜嬩欢鏁?0

[realtime] 鈿狅笍  鏁呴殰浜嬩欢妫€鍑?
  鏃堕棿: 45.32s - 48.76s (鎸佺画 3.44s)
  鏈€澶ф鐜? 0.892
  骞冲潎姒傜巼: 0.754
  鏁版嵁鐐规暟: 172

[receiver] 鐢垫満鍋滆浆瓒呰繃 5.0s锛岃嚜鍔ㄥ仠姝?[realtime] 妫€娴嬪畬鎴?  鎬诲抚鏁? 3823
  妫€娴嬫鏁? 368
  鏁呴殰浜嬩欢: 1
```

## 渚濊禆

閬ユ祴妯″潡闇€瑕侀澶栦緷璧栵細
```bash
pip install mavsdk pymavlink
```

## 娉ㄦ剰浜嬮」

1. **绔彛鍐茬獊** - 纭繚 UDP 绔彛 15550 鏈鍗犵敤
2. **MAVSDK 杩炴帴** - collector 榛樿杩炴帴 `udpin://0.0.0.0:14550`锛圥X4 GCS 绔彛锛?3. **鑷姩鍋滄** - receiver 鍦ㄧ數鏈哄仠杞?5 绉掑悗鑷姩鍋滄
4. **CSV 瑁佸壀** - receiver 浼氳嚜鍔ㄨ鎺夐檷钀藉悗鐨勫湴闈㈠抚
