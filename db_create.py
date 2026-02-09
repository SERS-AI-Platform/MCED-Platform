import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import os
import json

# ==========================================
# 1. DB 연결 설정
# ==========================================
DB_HOST = "localhost"
DB_NAME = "sers_db" # 미리 pgAdmin에서 database를 생성하거나 'postgres' 사용
DB_USER = "postgres"
DB_PASS = "solumhc1" 

# SQLAlchemy 엔진 생성 (Pandas 연동용)
connection_string = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"
engine = create_engine(connection_string)

# ==========================================
# 2. 테이블 생성 (Schema Init)
# ==========================================
def create_schema():
    commands = (
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id VARCHAR(50) PRIMARY KEY,
            group_name VARCHAR(50),
            age INTEGER,
            sex VARCHAR(10),
            height_cm FLOAT,
            weight_kg FLOAT,
            bmi FLOAT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS clinical_details (
            id SERIAL PRIMARY KEY,
            patient_id VARCHAR(50) REFERENCES patients(patient_id),
            diagnosis_date DATE,
            stage_clinical VARCHAR(50),
            tnm_t VARCHAR(20),
            tnm_n VARCHAR(20),
            tnm_m VARCHAR(20),
            medical_history TEXT,
            dietary_survey JSONB
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sers_samples (
            sample_uuid VARCHAR(100) PRIMARY KEY,
            patient_id VARCHAR(50) REFERENCES patients(patient_id),
            replicate_num INTEGER,
            raw_filename VARCHAR(255)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS spectral_data (
            point_id BIGSERIAL PRIMARY KEY,
            sample_uuid VARCHAR(100) REFERENCES sers_samples(sample_uuid),
            raman_shift_x FLOAT,
            intensity_y FLOAT
        )
        """
    )
    
    conn = None
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        cur = conn.cursor()
        for command in commands:
            cur.execute(command)
        cur.close()
        conn.commit()
        print("✅ 테이블 생성 완료")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

# ==========================================
# 3. 데이터 로더 (예시: 임상정보 & SERS)
# ==========================================

def load_clinical_data(filepath, group_name):
    """
    임상정보 엑셀 파일을 읽어 patients 테이블에 넣는 예시
    """
    df = pd.read_csv(filepath) # 또는 pd.read_excel(filepath)
    
    # 컬럼 매핑 (엑셀 헤더 -> DB 컬럼)
    # 실제 파일 헤더에 맞춰 수정 필요
    rename_map = {
        'SoluM Label': 'patient_id',
        '나이': 'age',
        '성별': 'sex',
        '신장': 'height_cm',
        '체중': 'weight_kg'
    }
    
    # 필요한 컬럼만 선택 및 이름 변경
    try:
        data_to_insert = df[rename_map.keys()].rename(columns=rename_map)
        data_to_insert['group_name'] = group_name
        
        # BMI 계산 (데이터에 없는 경우)
        # data_to_insert['bmi'] = data_to_insert['weight_kg'] / ((data_to_insert['height_cm']/100)**2)

        # 데이터 중복 방지 (이미 있으면 패스하는 로직은 추가 구현 필요, 여기선 append)
        data_to_insert.to_sql('patients', engine, if_exists='append', index=False, method='multi', chunksize=1000)
        print(f"✅ {group_name} 임상 데이터 업로드 완료: {len(data_to_insert)}건")
        
    except KeyError as e:
        print(f"❌ 컬럼 매핑 에러: {e}. 파일의 헤더를 확인해주세요.")

def load_sers_data(csv_filepath):
    """
    SERS CSV (X, Y 데이터)를 읽어 업로드
    파일명 예: CRC 1_1.CSV -> Patient: CRC 1, Replicate: 1
    """
    filename = os.path.basename(csv_filepath)
    name_part = os.path.splitext(filename)[0] # CRC 1_1
    
    # 파일명 파싱 로직
    if "_" in name_part:
        patient_id = name_part.rsplit('_', 1)[0] # CRC 1
        replicate = int(name_part.rsplit('_', 1)[1]) # 1
    else:
        print(f"⚠️ 파일명 형식 불일치: {filename}")
        return

    sample_uuid = f"{patient_id}_{replicate}" # Unique ID 생성
    
    # 1. Sample 정보 저장
    with engine.connect() as conn:
        conn.execute(
            "INSERT INTO sers_samples (sample_uuid, patient_id, replicate_num, raw_filename) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (sample_uuid, patient_id, replicate, filename)
        )

    # 2. 스펙트럼 데이터 저장 (대용량)
    # 헤더가 없는 경우 header=None, 컬럼명 지정
    df_spec = pd.read_csv(csv_filepath, header=None, names=['raman_shift_x', 'intensity_y'])
    df_spec['sample_uuid'] = sample_uuid
    
    # Bulk Insert
    df_spec.to_sql('spectral_data', engine, if_exists='append', index=False, method='multi', chunksize=10000)
    print(f"✅ SERS 데이터 업로드: {filename} ({len(df_spec)} points)")

# ==========================================
# 실행
# ==========================================
if __name__ == "__main__":
    create_schema()
    
    # 테스트용 실행 (경로 수정 필요)
    # load_clinical_data("path/to/LUN_data.csv", "Lung Cancer")
    # load_sers_data("path/to/CRC 1_1.CSV")