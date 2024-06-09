def all_icd_query(ds):
    """
    This method is optimized for All of Us platform. The code is coming from Tam's PheTK package with a few
    modifications to do what I want.

    It includes 3 queries: icd_query, v_icd_vocab_query, and final_query.
    icd_query retrieves all ICD codes from OMOP database.
    v_icd_vocab_query get the ICD codes starting with "V" from icd_query and check vocabulary_id using concept_id.
    final_query union distinct icd_query without V codes
    and v_icd_vocab_query which has V codes with proper vocabulary_ids.

    The reason for this is to ensure vocabulary_id values of V codes, many of which overlap between ICD9CM & ICD10CM,
    are correct.

    :param ds: Google BigQuery dataset ID containing OMOP data tables
    :return: a SQL query that would generate a table contains participant IDs and their ICD codes from unique dates
    """
    icd_query: str = f"""
        (
            SELECT DISTINCT
                co.person_id,
                co.condition_start_date AS date,
                c.vocabulary_id AS flag,
                c.concept_code AS ICD,
                co.condition_concept_id AS concept_id,
                DATE(p.birth_datetime) AS dob,
                DATE_DIFF(co.condition_start_date, DATE(p.birth_datetime), DAY) / 365.25 AS occurrence_age
            FROM
                {ds}.condition_occurrence AS co
            INNER JOIN
                {ds}.concept AS c
            ON
                co.condition_source_value = c.concept_code
            INNER JOIN
                {ds}.person AS p
            ON
                co.person_id = p.person_id
            WHERE
                c.vocabulary_id IN ('ICD9', 'ICD9CM', 'ICD10', 'ICD10CM')
        )
        UNION DISTINCT
        (
            SELECT DISTINCT
                co.person_id,
                co.condition_start_date AS date,
                c.vocabulary_id AS flag,
                c.concept_code AS ICD,
                co.condition_concept_id AS concept_id,
                DATE(p.birth_datetime) AS dob,
                DATE_DIFF(co.condition_start_date, DATE(p.birth_datetime), DAY) / 365.25 AS occurrence_age
            FROM
                {ds}.condition_occurrence AS co
            INNER JOIN
                {ds}.concept AS c
            ON
                co.condition_source_concept_id = c.concept_id
            INNER JOIN
                {ds}.person AS p
            ON
                co.person_id = p.person_id
            WHERE
                c.vocabulary_id IN ('ICD9CM', 'ICD10CM')
        )
        UNION DISTINCT
        (
            SELECT DISTINCT
                o.person_id,
                o.observation_date AS date,
                c.vocabulary_id AS flag,
                c.concept_code AS ICD,
                o.observation_concept_id AS concept_id,
                DATE(p.birth_datetime) AS dob,
                DATE_DIFF(o.observation_date, DATE(p.birth_datetime), DAY) / 365.25 AS occurrence_age
            FROM
                {ds}.observation AS o
            INNER JOIN
                {ds}.concept AS c
            ON
                o.observation_source_value = c.concept_code
            INNER JOIN
                {ds}.person AS p
            ON
                o.person_id = p.person_id
            WHERE
                c.vocabulary_id IN ('ICD9CM', 'ICD10CM')
        )
        UNION DISTINCT
        (
            SELECT DISTINCT
                o.person_id,
                o.observation_date AS date,
                c.vocabulary_id AS flag,
                c.concept_code AS ICD,
                o.observation_concept_id AS concept_id,
                DATE(p.birth_datetime) AS dob,
                DATE_DIFF(o.observation_date, DATE(p.birth_datetime), DAY) / 365.25 AS occurrence_age
            FROM
                {ds}.observation AS o
            INNER JOIN
                {ds}.concept AS c
            ON
                o.observation_source_concept_id = c.concept_id
            INNER JOIN
                {ds}.person AS p
            ON
                o.person_id = p.person_id
            WHERE
                c.vocabulary_id IN ('ICD9CM', 'ICD10CM')
        )
        """

    v_icd_vocab_query: str = f"""
            SELECT DISTINCT
                v_icds.person_id,
                v_icds.date,
                v_icds.ICD,
                c.vocabulary_id AS flag,
                v_icds.occurrence_age
            FROM
                (
                    SELECT
                        *
                    FROM
                        ({icd_query}) AS icd_events
                    WHERE
                        icd_events.ICD LIKE "V%"
                ) AS v_icds
            INNER JOIN
                {ds}.concept_relationship AS cr
            ON
                v_icds.concept_id = cr.concept_id_1
            INNER JOIN
                {ds}. concept AS c
            ON
                cr.concept_id_2 = c.concept_id
            WHERE
                c.vocabulary_id IN ("ICD9CM", "ICD10CM")
            AND
                v_icds.ICD = c.concept_code
            AND NOT
                v_icds.flag != c.vocabulary_id
        """

    final_query: str = f"""
            (
                SELECT DISTINCT
                    person_id,
                    date,
                    ICD,
                    flag,
                    occurrence_age
                FROM 
                    ({icd_query})
                WHERE
                    NOT ICD LIKE "V%"
            )
            UNION DISTINCT
            (
                SELECT DISTINCT
                    *
                FROM
                    ({v_icd_vocab_query})
            )
        """
    return final_query


def all_demo_query(ds):
    """
    This method is optimized for the All of Us platform. This query generates a table containing participant IDs and
    their ICD codes from unique dates
    :param ds: Google BigQuery dataset ID containing OMOP data tables
    :return: a SQL query that would generate a table contains participant IDs and their demographic characteristics
    """

    demo_query: str = f"""
        SELECT 
            demos.person_id, 
            demos.sex, 
            demos.dob, 
            ages.first_date, 
            ages.last_date,
            DATE_DIFF(ages.first_date, demos.dob, DAY) / 365.25 AS first_age,
            DATE_DIFF(ages.last_date, demos.dob, DAY) / 365.25 AS last_age
        FROM 
            (
                SELECT 
                    icds.person_id, 
                    MIN(icds.entry_date) AS first_date, 
                    MAX(icds.entry_date) AS last_date
                FROM 
                    (
                        SELECT DISTINCT
                            p.person_id, 
                            DATE(e.condition_start_date) AS entry_date
                        FROM 
                            {ds}.condition_occurrence e
                        JOIN 
                            {ds}.person p ON p.person_id = e.person_id
                        JOIN 
                            {ds}.concept co ON co.concept_id = e.condition_source_concept_id
                        WHERE 
                            co.vocabulary_id IN ('ICD9CM', 'ICD10CM')
                        GROUP BY 
                            p.person_id, e.condition_start_date
                    ) AS icds
                GROUP BY 
                    icds.person_id
            ) AS ages
        JOIN 
            (
                SELECT 
                    person_id, 
                    c1.concept_name AS sex, 
                    DATE(birth_datetime) AS dob 
                FROM 
                    {ds}.person p1
                JOIN 
                    {ds}.concept c1 ON c1.concept_id = p1.sex_at_birth_concept_id
            ) AS demos ON ages.person_id = demos.person_id;
       
        """

    return demo_query
