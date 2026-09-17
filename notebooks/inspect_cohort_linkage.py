from pathlib import Path
import openpyxl,json
r=Path(__file__).resolve().parent.parent
for path in [r/'data/01_clinical_metadata/hospital_clinical_tables/보라매 병원 임상정보.xlsx',r/'data/03_sers_date_lot_balanced_acquisition/thermo_mapping_BNOR-BPRO_20260810-20260814/clinical_df.xlsx']:
 w=openpyxl.load_workbook(path,data_only=True,read_only=True);rows=w.active.iter_rows(values_only=True);h=next(rows);print(path.name,[str(x) for x in h]);w.close()
