# RflyMAD Processing Execution Plan  
  
Core rules  
- source raw is read-only.  
- working raw is writable and stores generated topic CSV files.  
- processed output, manifest, and logs stay outside raw folders.  
- processed, TestInfo metadata, and case_mapping must come from the same run.  
- do not merge old processed with new raw or new TestInfo.  
  
Recommended directories  
- D:\RflyMAD : source raw  
- D:\RflyMAD_work : working raw  
- D:\UAV-Data\processed_official_50hz : processed Case_*.csv  
- D:\UAV-Data\manifest : manifest_v2 and TestInfo metadata  
- D:\UAV-Data\logs : failed_cases.log and case_mapping.csv  
  
Confirmed pitfall  
- SIL and HIL case_num depends on directory enumeration order.  
- the adapted tool uses os.listdir and loop index to build caseID and case_num.  
- old processed is not a stable merge baseline for a new TestInfo round.  
  
case_mapping.csv  
- required fields: processed_case_id, processed_csv_path, platform, flight_mode, fault_type_canonical, raw_case_dir, raw_case_path, testinfo_path, raw_case_index_in_dir, build_batch_id.  
- stable identity should use platform, flight_mode, fault_type_canonical, and raw_case_dir.  
- bucket_case_num stays only as fallback.  
  
Merge order  
- run HIL first, Real second, SIL later.  
- build manifest_v2 Stage 1 from processed CSV first.  
- align TestInfo metadata with case_mapping next.  
- run Stage 2 enrichment last.  
  
Immediate next step  
- refactor the entry script to separate source raw, working raw, processed output, and logs.  
- emit case_mapping.csv during processing.  
- validate the chain with one small HIL smoke test. 
