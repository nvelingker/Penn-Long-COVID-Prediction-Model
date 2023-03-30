from global_code import *

def personal_notes(note_nlp, note):

    person_notes_df = note.select('person_id', 'note_id', 'note_date', 'visit_occurrence_id')
    note_concept_df = note_nlp.select('note_nlp_id', 'note_id', 'term_modifier_certainty', 'note_nlp_concept_id', 'note_nlp_concept_name', )
    df = person_notes_df.join(note_concept_df, 'note_id', 'left')

    return df

def personal_notes_pos_neg(personal_notes):
    positive_mask = personal_notes['term_modifier_certainty'].isin(['Positive'])
    negative_mask = personal_notes['term_modifier_certainty'].isin(['Negated'])
    pos_neg_mask = personal_notes['term_modifier_certainty'].isin(['Positive', 'Negated'])

    positive_personal_notes = personal_notes.filter(positive_mask)
    negative_personal_notes = personal_notes.filter(negative_mask)
    pos_neg_personal_notes = personal_notes.filter(pos_neg_mask)

    note_with_both_pos_neg = positive_personal_notes.alias('a').join(negative_personal_notes.alias('b'), (F.col("a.note_id") == F.col("b.note_id")) & (F.col("a.note_nlp_concept_id") == F.col("b.note_nlp_concept_id")), "inner").select(F.col("a.note_id"),F.col("a.note_nlp_concept_id"))

    df = pos_neg_personal_notes.alias('a').join(note_with_both_pos_neg.alias('b'), (F.col("a.note_id") == F.col("b.note_id")) & (F.col("a.note_nlp_concept_id") == F.col("b.note_nlp_concept_id")), "left_outer")\
                 .where(F.col("b.note_id").isNull() & F.col("b.note_nlp_concept_id").isNull())\
                 .select([F.col(f"a.{c}") for c in pos_neg_personal_notes.columns]).distinct()

    return df

def related_concept(concept_set_members, broad_related_concepts):
    related_concepts = broad_related_concepts

    concept_set_members_df = concept_set_members.select("codeset_id",  "concept_id", "concept_name",)
    related_concepts_df = related_concepts.select("codeset_id", "concept_set_name",)
    df = concept_set_members_df.join(related_concepts_df, "codeset_id", 'right')
    return df

def personal_symptom(personal_notes_pos_neg, related_concept):

    personal_notes_pos_neg_df = personal_notes_pos_neg.select('*')
    related_concept_df = related_concept.select('*')
    df = personal_notes_pos_neg_df.join(related_concept_df, personal_notes_pos_neg_df.note_nlp_concept_id == related_concept_df.concept_id, 'inner')

    return df

def positive_symptoms(personal_symptom):

    rslt_df = personal_symptom[personal_symptom['term_modifier_certainty'] == "Positive"]
    return rslt_df

def nlp_sym_analysis(positive_symptoms, Long_COVID_Silver_Standard):
    TABLE = positive_symptoms
    CONCEPT_NAME_COL = "concept_name"
    l, h = 0, 1000


    label = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    TABLE = TABLE.join(label, "person_id").select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome")).filter(F.col(CONCEPT_NAME_COL) != "No matching concept")
    distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(h).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]

    pos, count = [], []
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    print(len(distinct))
    t = time.time()
    for cname in distinct[l:]:
        f = TABLE.agg(
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1))
        ).collect()
        one_count = f[0][1]
        size = f[0][0]
        pos.append(one_count/size)
        count.append(size)
    print(time.time() - t)
    r = pd.DataFrame(list(zip(distinct,pos, count)), columns=[CONCEPT_NAME_COL,"pos", "count"])
    r['neg'] = r.apply(lambda row: 1-row.pos, axis = 1)
    r['max'] = r.apply(lambda row: max(row.pos, 1-row.pos), axis=1)
    r =r[[CONCEPT_NAME_COL,'pos','neg','max','count']]

    return r

def top_concept_names(nlp_sym_analysis):
    r = nlp_sym_analysis.withColumn("scale_above_threshold", F.when((F.col("max") > 0.65), F.col("max")*F.col("count")).otherwise(F.lit(0)))
    return r

def important_concepts(positive_symptoms, top_concept_names):

    names = top_concept_names[(top_concept_names["scale_above_threshold"]) > 0]
    names = names[(names["count"] > 200)][["concept_name", "count", "scale_above_threshold"]]
    concept_ids = positive_symptoms[["concept_name", "concept_id"]]
    concept_ids = concept_ids.drop_duplicates()
    df = pd.merge(names, concept_ids, "inner", "concept_name")

    return df


def person_nlp_symptom(personal_symptom, broad_related_concepts):

    personal_symptom_df = personal_symptom[['note_id', 'person_id', 'note_date', 'visit_occurrence_id']]
    personal_symptom_df = personal_symptom_df.drop_duplicates()
    personal_symptom_df = personal_symptom_df.set_index('note_id')
    all_symptoms_df = broad_related_concepts
    all_symptoms = list(set(all_symptoms_df["concept_set_name"]))

    print(all_symptoms)

    for symptom in all_symptoms:
        symptom_column_name = "sympt_" + symptom.replace("-", "_")
        new_symptom_df = personal_symptom[['note_id', 'term_modifier_certainty', 'concept_set_name']]
        new_symptom_df = new_symptom_df.loc[new_symptom_df.concept_set_name == symptom]
        new_symptom_df = new_symptom_df.rename(columns={'term_modifier_certainty': symptom_column_name})
        new_symptom_df = new_symptom_df[["note_id", symptom_column_name]]
        new_symptom_df = new_symptom_df.drop_duplicates()

        personal_symptom_df = personal_symptom_df.merge(new_symptom_df, on="note_id", how="left")
        personal_symptom_df[symptom_column_name] = personal_symptom_df[symptom_column_name].map(lambda x: 1 if x == 'Positive' else (-1 if x == 'Negated' else np.nan))

    return personal_symptom_df

def important_concepts(positive_symptoms, top_concept_names):

    names = top_concept_names[(top_concept_names["scale_above_threshold"]) > 0]
    names = names[(names["count"] > 200)][["concept_name", "count", "scale_above_threshold"]]
    concept_ids = positive_symptoms[["concept_name", "concept_id"]]
    concept_ids = concept_ids.drop_duplicates()
    df = pd.merge(names, concept_ids, "inner", "concept_name")

    return df

def person_top_nlp_symptom(personal_symptom, important_concepts):

    personal_symptom_df = personal_symptom[['note_id', 'person_id', 'note_date', 'visit_occurrence_id']]
    personal_symptom_df = personal_symptom_df.drop_duplicates()
    personal_symptom_df = personal_symptom_df.set_index('note_id')
    all_symptoms_df = important_concepts
    all_symptoms = list(set(all_symptoms_df["concept_name"]))

    print(all_symptoms)

    for symptom in all_symptoms:
        symptom_column_name = symptom.replace(' ', '_').replace('-', '_').replace(',', '').replace('(', '').replace(')', '')
        new_symptom_df = personal_symptom[['note_id', 'term_modifier_certainty', 'concept_name']]

        new_symptom_df = new_symptom_df.loc[new_symptom_df.concept_name == symptom]
        new_symptom_df = new_symptom_df.rename(columns={'term_modifier_certainty': symptom_column_name})
        new_symptom_df = new_symptom_df[["note_id", symptom_column_name]]
        new_symptom_df = new_symptom_df.drop_duplicates()

        personal_symptom_df = personal_symptom_df.merge(new_symptom_df, on="note_id", how="left")
        personal_symptom_df[symptom_column_name] = personal_symptom_df[symptom_column_name].map(lambda x: 1 if x == 'Positive' else (-1 if x == 'Negated' else np.nan))

    return personal_symptom_df
