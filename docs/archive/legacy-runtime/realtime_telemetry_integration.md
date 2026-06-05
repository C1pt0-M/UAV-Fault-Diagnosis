# 瀹炴椂閬ユ祴涓庢晠闅滄娴?鈥?鍓嶅悗绔泦鎴愯璁?
## 1. 鍔熻兘姒傝堪

鍦?Dashboard 涓泦鎴愬疄鏃堕仴娴嬮噰闆嗕笌鏁呴殰妫€娴嬪姛鑳斤紝鏀寔锛?
1. **瀹炴椂鏁版嵁閲囬泦** - 鍚姩 PX4 SITL + 閬ユ祴閲囬泦锛屾暟鎹瓨鍏ユ暟鎹簱
2. **瀹炴椂鏁呴殰妫€娴?* - 鍙€夊惎鍔ㄥ湪绾挎娴嬶紝瀹炴椂鎺ㄩ€佸憡璀?3. **鏃ュ織绠＄悊** - 閲囬泦璁板綍鍦ㄦ棩蹇楃鐞嗕腑鍙煡鐪?4. **鐘舵€佺洃鎺?* - 鍓嶇瀹炴椂鏄剧ず閲囬泦/妫€娴嬬姸鎬?
## 2. 绯荤粺鏋舵瀯

### 2.1 鍚庣鏋舵瀯

```
Flask API
  鈹溾攢鈹€ /api/telemetry/start-collection    # 鍚姩鏁版嵁閲囬泦
  鈹溾攢鈹€ /api/telemetry/stop-collection     # 鍋滄鏁版嵁閲囬泦
  鈹溾攢鈹€ /api/telemetry/start-detection     # 鍚姩瀹炴椂妫€娴?  鈹溾攢鈹€ /api/telemetry/stop-detection      # 鍋滄瀹炴椂妫€娴?  鈹溾攢鈹€ /api/telemetry/status              # 鑾峰彇杩愯鐘舵€?  鈹斺攢鈹€ /api/telemetry/sessions            # 鑾峰彇閲囬泦浼氳瘽鍒楄〃

鍚庣鏈嶅姟
  鈹溾攢鈹€ TelemetryProcessManager            # 杩涚▼绠＄悊鏈嶅姟
  鈹?  鈹溾攢鈹€ PX4 SITL 杩涚▼
  鈹?  鈹溾攢鈹€ Collector 杩涚▼锛堟棤浜烘満渚ч噰闆嗭級
  鈹?  鈹溾攢鈹€ Receiver 杩涚▼锛堜簯绔帴鏀讹級
  鈹?  鈹斺攢鈹€ Detector 杩涚▼锛堝疄鏃舵娴嬶紝鍙€夛級
  鈹斺攢鈹€ TelemetryDataService               # 鏁版嵁鏈嶅姟
      鈹溾攢鈹€ CSV 鈫?鏁版嵁搴撳鍏?      鈹溾攢鈹€ 瀹炴椂妫€娴嬬粨鏋滆褰?      鈹斺攢鈹€ 鏃ュ織鐢熸垚
```

### 2.2 鏁版嵁搴撹璁?
#### 鏂板琛細telemetry_sessions

```sql
CREATE TABLE telemetry_sessions (
    id VARCHAR(20) PRIMARY KEY,           -- 浼氳瘽ID: TS-XXXXXX
    drone_id VARCHAR(20),                 -- 鍏宠仈鏃犱汉鏈?    session_type VARCHAR(20),             -- 绫诲瀷: simulation/real
    start_time DATETIME,                  -- 寮€濮嬫椂闂?    end_time DATETIME,                    -- 缁撴潫鏃堕棿
    duration_seconds FLOAT,               -- 鎸佺画鏃堕暱
    frame_count INTEGER,                  -- 鎬诲抚鏁?    csv_path TEXT,                        -- CSV 鏂囦欢璺緞
    has_detection BOOLEAN,                -- 鏄惁鍚敤妫€娴?    fault_events_count INTEGER,           -- 妫€鍑烘晠闅滀簨浠舵暟
    status VARCHAR(20),                   -- 鐘舵€? running/completed/failed
    created_by VARCHAR(50),               -- 鍒涘缓浜?    created_at DATETIME,
    FOREIGN KEY (drone_id) REFERENCES drones(id)
);
```

#### 鏂板琛細telemetry_fault_events

```sql
CREATE TABLE telemetry_fault_events (
    id VARCHAR(20) PRIMARY KEY,           -- 浜嬩欢ID: FE-XXXXXX
    session_id VARCHAR(20),               -- 鍏宠仈浼氳瘽
    start_time FLOAT,                     -- 浜嬩欢寮€濮嬫椂闂达紙鐩稿绉掞級
    end_time FLOAT,                       -- 浜嬩欢缁撴潫鏃堕棿
    duration FLOAT,                       -- 鎸佺画鏃堕暱
    max_fault_prob FLOAT,                 -- 鏈€澶ф晠闅滄鐜?    mean_fault_prob FLOAT,                -- 骞冲潎鏁呴殰姒傜巼
    data_points INTEGER,                  -- 鏁版嵁鐐规暟
    severity VARCHAR(20),                 -- 涓ラ噸搴? low/medium/high/critical
    created_at DATETIME,
    FOREIGN KEY (session_id) REFERENCES telemetry_sessions(id)
);
```

#### 鎵╁睍鐜版湁 logs 琛?
鏂板鏃ュ織绫诲瀷锛?- `telemetry` - 閬ユ祴閲囬泦鏃ュ織
- `realtime_detection` - 瀹炴椂妫€娴嬫棩蹇?

## 3. API 鎺ュ彛瑙勮寖

### 3.1 鍚姩鏁版嵁閲囬泦

**POST** `/api/telemetry/start-collection`

璇锋眰浣擄細
```json
{
  "drone_id": "DRONE-001",
  "session_type": "simulation",
  "enable_detection": false
}
```

鍝嶅簲锛?```json
{
  "code": 200,
  "message": "鏁版嵁閲囬泦宸插惎鍔?,
  "data": {
    "session_id": "TS-123456",
    "status": "running",
    "processes": {
      "px4_sitl": "running",
      "collector": "running",
      "receiver": "running",
      "detector": null
    }
  }
}
```

### 3.2 鍋滄鏁版嵁閲囬泦

**POST** `/api/telemetry/stop-collection`

鍝嶅簲锛?```json
{
  "code": 200,
  "message": "鏁版嵁閲囬泦宸插仠姝?,
  "data": {
    "session_id": "TS-123456",
    "duration_seconds": 125.3,
    "frame_count": 6265,
    "csv_path": "/path/to/telemetry_TS-123456.csv"
  }
}
```

### 3.3 鍚姩瀹炴椂妫€娴?
**POST** `/api/telemetry/start-detection`

鍝嶅簲锛?```json
{
  "code": 200,
  "message": "瀹炴椂妫€娴嬪凡鍚姩",
  "data": {
    "detector_status": "running"
  }
}
```

### 3.4 鑾峰彇杩愯鐘舵€?
**GET** `/api/telemetry/status`

鍝嶅簲锛?```json
{
  "code": 200,
  "data": {
    "is_collecting": true,
    "is_detecting": false,
    "current_session_id": "TS-123456",
    "uptime_seconds": 45.2,
    "frame_count": 2260,
    "fault_events_count": 0,
    "processes": {
      "px4_sitl": "running",
      "collector": "running",
      "receiver": "running",
      "detector": null
    }
  }
}
```

### 3.5 鑾峰彇浼氳瘽鍒楄〃

**GET** `/api/telemetry/sessions?page=1&per_page=10`

鍝嶅簲锛?```json
{
  "code": 200,
  "data": {
    "items": [
      {
        "id": "TS-123456",
        "drone_id": "DRONE-001",
        "start_time": "2026-03-15 23:30:00",
        "duration_seconds": 125.3,
        "frame_count": 6265,
        "has_detection": true,
        "fault_events_count": 2,
        "status": "completed"
      }
    ],
    "total": 15,
    "page": 1,
    "per_page": 10
  }
}
```

## 4. 鍓嶇浜や簰娴佺▼

### 4.1 椤甸潰鍏ュ彛

鍦?Dashboard 渚ц竟鏍忔柊澧炴垨澶嶇敤鐜版湁鑿滃崟椤癸紝杩涘叆"瀹炴椂閬ユ祴"椤甸潰銆?
### 4.2 鐢ㄦ埛鎿嶄綔娴佺▼

```
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?瀹炴椂閬ユ祴                                                    鈹?鈹?                                                            鈹?鈹? 鐘舵€? 鈼?绌洪棽                    鏃犱汉鏈? [DRONE-001 鈻糫     鈹?鈹?                                                            鈹?鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                       鈹?鈹? 鈹?鈻?鍚姩閲囬泦   鈹? 鈹?馃攳 鍚姩妫€娴? 鈹?(閲囬泦杩愯涓墠鍙偣鍑?   鈹?鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                       鈹?鈹?                                                            鈹?鈹? 鈹€鈹€ 杩愯涓樉绀?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€   鈹?鈹? 宸查噰闆? 2260 甯?| 杩愯: 45.2s | 妫€娴? 鍏抽棴               鈹?鈹? 杩涚▼: PX4[鉁揮 Collector[鉁揮 Receiver[鉁揮 Detector[鉁梋        鈹?鈹?                                                            鈹?鈹? 鈹€鈹€ 妫€娴嬪憡璀︼紙鍚敤妫€娴嬪悗鏄剧ず锛夆攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€    鈹?鈹? 鈿?45.32s-48.76s  鏁呴殰姒傜巼 89.2%  涓ラ噸搴? high            鈹?鈹?                                                            鈹?鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                                         鈹?鈹? 鈹?鈻?鍋滄閲囬泦   鈹?                                         鈹?鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                                         鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?```

### 4.3 鐘舵€佹満

```
绌洪棽 鈹€鈹€[鍚姩閲囬泦]鈹€鈹€鈫?閲囬泦涓?鈹€鈹€[鍚姩妫€娴媇鈹€鈹€鈫?閲囬泦+妫€娴嬩腑
                       鈹?                       鈹?                       鈹?[鍋滄閲囬泦]         [鍋滄妫€娴媇
                       鈫?                       鈹?                    瀹屾垚/鍏ュ簱 鈫愨攢鈹€[鍋滄閲囬泦]鈹€鈹€鈹€鈹€鈹€鈹€鈹?```

- **绌洪棽**: 鍙偣鍑?鍚姩閲囬泦"
- **閲囬泦涓?*: 鍙偣鍑?鍚姩妫€娴?鎴?鍋滄閲囬泦"
- **閲囬泦+妫€娴嬩腑**: 鍙偣鍑?鍋滄妫€娴?鎴?鍋滄閲囬泦"锛堝仠姝㈤噰闆嗕細鍚屾椂鍋滄妫€娴嬶級
- **瀹屾垚**: 鏁版嵁鍐欏叆鏁版嵁搴擄紝鍦ㄦ棩蹇楃鐞嗕腑鍙煡

### 4.4 鍓嶇杞绛栫暐

閲囬泦杩愯鏈熼棿锛屽墠绔瘡 **2 绉?* 璋冪敤 `GET /api/telemetry/status` 鑾峰彇鏈€鏂扮姸鎬侊紙甯ф暟銆佹娴嬩簨浠剁瓑锛夈€傚仠姝㈠悗鍋滄杞銆?
## 5. 杩涚▼鐢熷懡鍛ㄦ湡

### 5.1 鍚姩閲囬泦锛坰tart-collection锛?
```
1. 鍒涘缓 TelemetrySession 璁板綍锛坰tatus=running锛?2. 鍐欏叆鏃ュ織锛坱ype=telemetry, "寮€濮嬮仴娴嬮噰闆?锛?3. 鍚姩 PX4 SITL 瀛愯繘绋?4. 绛夊緟 PX4 灏辩华锛堟娴?TCP 18570 绔彛锛?5. 鍚姩 Receiver 绾跨▼锛堢洃鍚?UDP 15550锛屽啓 CSV锛?6. 鍚姩 Collector 瀛愯繘绋嬶紙杩炴帴 PX4锛?0Hz 杞彂 UDP锛?```

### 5.2 鍚姩妫€娴嬶紙start-detection锛岄渶瑕佸厛鍚姩閲囬泦锛?
```
1. 灏?FaultDetector + FaultEventAggregator 娉ㄥ叆 Receiver 鐨?callback
2. 姣忓抚鍒拌揪鍚庢鏌ョ獥鍙ｆ槸鍚︽弧 150 甯э紝婊″垯鎺ㄧ悊
3. 妫€鍑烘晠闅滀簨浠舵椂鍐欏叆 telemetry_fault_events 琛?4. 鏇存柊 session.has_detection = True
```

### 5.3 鍋滄閲囬泦锛坰top-collection锛?
```
1. 濡傛灉妫€娴嬪湪杩愯锛屽厛鍋滄妫€娴?2. 鍋滄 Collector 杩涚▼锛圫IGTERM锛?3. 鍋滄 Receiver 绾跨▼锛堣缃?stop flag锛?4. 鍋滄 PX4 SITL 杩涚▼锛圫IGTERM 鈫?绛夊緟 鈫?SIGKILL锛?5. 鏇存柊 session锛坋nd_time, duration, frame_count, status=completed锛?6. 鍐欏叆鏃ュ織锛坱ype=telemetry, "閬ユ祴閲囬泦瀹屾垚锛屽叡 N 甯?锛?```

## 6. 鏃ュ織绠＄悊闆嗘垚

閲囬泦鍜屾娴嬩骇鐢熺殑鏃ュ織浣跨敤鐜版湁 Log 妯″瀷锛岃嚜鍔ㄥ啓鍏ヤ互涓嬬被鍨嬶細

| 鏃舵満 | type | content 绀轰緥 |
|------|------|-------------|
| 鍚姩閲囬泦 | telemetry | 寮€濮嬮仴娴嬫暟鎹噰闆嗭紝浼氳瘽 TS-123456 |
| 鍋滄閲囬泦 | telemetry | 閬ユ祴閲囬泦瀹屾垚锛屽叡 6265 甯э紝鏃堕暱 125.3s |
| 鍚姩妫€娴?| realtime_detection | 鍚姩瀹炴椂鏁呴殰妫€娴嬶紝鍏宠仈浼氳瘽 TS-123456 |
| 妫€鍑烘晠闅?| realtime_detection | 妫€鍑烘晠闅滀簨浠?45.3s-48.8s锛屾渶澶ф鐜?89.2% |
| 鍋滄妫€娴?| realtime_detection | 鍋滄瀹炴椂妫€娴嬶紝鍏辨鍑?2 涓晠闅滀簨浠?|
| 閲囬泦澶辫触 | telemetry | 閬ユ祴閲囬泦寮傚父缁堟: PX4 杩涚▼閫€鍑?|

鍦ㄦ棩蹇楃鐞嗛〉闈腑锛岀敤鎴峰彲鎸?`type=telemetry` 鎴?`type=realtime_detection` 绛涢€夋煡鐪嬨€?
## 7. 瀹炵幇瑕佺偣

### 7.1 鍚庣鏂囦欢娓呭崟

| 鏂囦欢 | 璇存槑 |
|------|------|
| `app/assets/models/telemetry_session.py` | TelemetrySession + TelemetryFaultEvent 妯″瀷 |
| `app/services/telemetry_manager.py` | TelemetryProcessManager 杩涚▼绠＄悊 |
| `app/routes/telemetry.py` | API 璺敱钃濆浘 |
| `src/telemetry/` | 閬ユ祴閲囬泦妯″潡锛堝凡闆嗘垚锛?|
| `src/online/realtime_detection_service.py` | 瀹炴椂妫€娴嬫湇鍔★紙宸插疄鐜帮級 |

### 7.2 鍏抽敭璁捐鍐崇瓥

1. **杩涚▼绠＄悊鐢?subprocess** 鈥?PX4 SITL 鍜?Collector 浣滀负鐙珛瀛愯繘绋嬬鐞嗭紝Receiver 鍜?Detector 鍦?Flask 杩涚▼鍐呬互绾跨▼杩愯锛堝叡浜唴瀛橈紝鏂逛究鍥炶皟娉ㄥ叆锛?2. **鍗曚緥妯″紡** 鈥?TelemetryProcessManager 鍏ㄥ眬鍞竴锛屽悓涓€鏃堕棿鍙厑璁镐竴涓噰闆嗕細璇?3. **CSV 瀛樺偍璺緞** 鈥?`data/telemetry_sessions/{session_id}.csv`
4. **妫€娴嬪彲鐑彃鎷?* 鈥?閲囬泦杩愯涓彲闅忔椂鍚姩/鍋滄妫€娴嬶紝涓嶅奖鍝嶆暟鎹噰闆?
### 7.3 鍓嶇瀹炵幇寤鸿锛堜緵鍓嶇鍙傝€冿級

1. 鏂板缓 `src/views/Telemetry.vue` 鎴栧湪鐜版湁椤甸潰涓坊鍔犲叆鍙?2. 浣跨敤 `setInterval` 姣?2s 杞 status 鎺ュ彛
3. 鏁呴殰鍛婅鐢?`el-alert` 鎴?`notification` 缁勪欢瀹炴椂灞曠ず
4. 杩涚▼鐘舵€佺敤 `el-tag` 鏍囩鏄剧ず锛堢豢鑹?running / 鐏拌壊 stopped锛?5. 浼氳瘽鍘嗗彶鍒楄〃澶嶇敤鐜版湁 Table 缁勪欢锛屾敮鎸佽烦杞埌鏃ュ織璇︽儏