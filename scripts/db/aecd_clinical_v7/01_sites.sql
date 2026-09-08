-- Purpose:
--   Make master.sites hold one row per collecting hospital.
--
-- Why the rename:
--   site_id 1 was created as 'smcxd07', which is a *protocol* code, not a site.
--   A hospital runs several protocols (CBNUH spans SMCXD01/SMCXD06/SMCMD06 and
--   one unknown), so protocol cannot identify a site. site_id is unchanged, so
--   every existing foreign key (master.subjects, and through it measurement.*)
--   keeps pointing at the same row.
--
-- Expected result:
--   sites = 8 (1 renamed + 7 inserted).

BEGIN;

DO $$
BEGIN
    IF current_database() <> 'aecd_platform' THEN
        RAISE EXCEPTION 'Wrong database: %.', current_database();
    END IF;
END
$$;

UPDATE master.sites
SET
    site_code = 'BORAMAE',
    site_name = 'Seoul National University Boramae Hospital'
WHERE site_code = 'smcxd07';

INSERT INTO master.sites (site_code, site_name)
VALUES
    ('CBNUH',   'Chungbuk National University Hospital'),
    ('IJBPH',   'Inje University Busan Paik Hospital'),
    ('SNUH',    'Seoul National University Hospital'),
    ('SSMH',    'Seoul St. Mary''s Hospital'),
    ('SAMSUNG', 'Samsung Seoul Hospital'),
    ('YONSEI',  'Yonsei Severance Hospital'),
    ('YPNUH',   'Yangsan Pusan National University Hospital')
ON CONFLICT (site_code) DO NOTHING;

-- Keep the 2026-08 staging rows joinable: they were loaded with the old code.
UPDATE ingest.clinical_master_staging
SET site_code = 'BORAMAE'
WHERE btrim(site_code) = 'smcxd07';

COMMIT;

SELECT site_id, site_code, site_name
FROM master.sites
ORDER BY site_id;
