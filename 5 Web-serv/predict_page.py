import pandas as pd
import numpy as np
import streamlit as st
import dill as pickle



def show_predict_page():

    cat = None

    def load_model():
        out = []

        #классификация
        # with open(r'C:\Users\admin\Desktop\PYTHON_2022\python_study_different_not_for_git\sber_finall\models\saved_model.pkl',
        #           'rb') as file:
        #     out.append(pickle.load(file))

        with open(r'C:\Users\admin\Desktop\PYTHON_2022\python_study_different_not_for_git\sber_finall\models\saved_model_cb.pkl',
                  'rb') as file:
            out.append(pickle.load(file))

        #регрессия
        with open(r'C:\Users\admin\Desktop\PYTHON_2022\python_study_different_not_for_git\sber_finall\models\saved_model_regr_cb.pkl',
                  'rb') as file:
            out.append(pickle.load(file))

        # with open(r'C:\Users\admin\Desktop\PYTHON_2022\python_study_different_not_for_git\sber_finall\models\saved_model_regr.pkl',
        #           'rb') as file:
        #     out.append(pickle.load(file))

        return out

    load = load_model()
    data = load[0]
    model_loaded = data["model"]
    vectorize_columns_loaded = data["vectorize_columns"]

    def category_predict(name, skills, description, experience, schedule,
                         model_loaded=model_loaded,
                         vectorize_columns_loaded=vectorize_columns_loaded):

        df_sample = pd.DataFrame([[0] * 8], columns=['Experience_Более 6 лет', 'Experience_Нет опыта',
                                                     'Experience_От 1 года до 3 лет', 'Experience_От 3 до 6 лет',
                                                     'Schedule_Гибкий график', 'Schedule_Полный день',
                                                     'Schedule_Сменный график', 'Schedule_Удаленная работа'])

        df_sample = set_dummies_exp(df_sample, experience)
        df_sample = set_dummies_sched(df_sample, schedule)

        df_for_predict = pd.concat([pd.DataFrame([[name, skills, description]],
                                                 columns=['Name', 'Skills', 'Description']), df_sample], axis=1)
        tokenized_columns = pd.DataFrame.sparse.from_spmatrix(vectorize_columns_loaded.transform(df_for_predict))
        df_for_predict_new = pd.concat([tokenized_columns, df_for_predict.iloc[:, 3:].reset_index(drop=True)], axis=1)

        y_pred = model_loaded.predict(df_for_predict_new)

        def y_pred_to_cat(y_pred):
            if y_pred == 1:
                return 'Data Science'
            elif y_pred == 2:
                return 'Data Analysis'
            elif y_pred == 3:
                return 'Data Engineer'
            else:
                return 'Python Develop'

        return y_pred_to_cat(y_pred)

    def set_dummies_exp(df, experience):
      if experience == 'Более 6 лет':
        df['Experience_Более 6 лет'] = 1
      elif experience == 'От 3 до 6 лет':
        df['Experience_От 3 до 6 лет'] = 1
      elif experience == 'От 1 года до 3 лет':
        df['Experience_От 1 года до 3 лет'] = 1
      else:
        df['Experience_Нет опыта'] = 1

      return df

    def set_dummies_sched(df, schedule):
      if schedule == 'Полный день':
        df['Schedule_Полный день'] = 1
      elif schedule == 'Сменный график':
        df['Schedule_Сменный график'] = 1
      elif schedule == 'Удаленная работа':
        df['Schedule_Удаленная работа'] = 1
      else:
        df['Schedule_Гибкий график'] = 1

      return df

    st.write('''Заполните следующую форму для получения прогноза''')

    experiences = (' ', 'Более 6 лет', 'Нет опыта', 'От 1 года до 3 лет', 'От 3 до 6 лет')

    schedules = (' ','Гибкий график', 'Полный день', 'Сменный график', 'Удаленная работа')

    with st.form(key='experiene_shdule'):

        name = st.text_input('Должность').replace('\\',' ').replace('/', ' ').replace('(', ' ').replace(')', ' ')
        skills = st.text_area('Навыки').replace("\n", ",")
        description = st.text_area('Описание вакансии')
        col1, col2 = st.columns([1, 1])
        with col1:
            experience = st.selectbox('Опыт работы', experiences)
            salary_predict_trigger = st.form_submit_button('Сделать прогноз по зарплате')
        with col2:
            schedule = st.selectbox('График работы', schedules)
            category_predict_trigger = st.form_submit_button('Классификация по категориям')

    if experience == ' ':
        experience = 'От 1 года до 3 лет'

    if schedule == ' ':
        schedule = 'Удаленная работа'

    if category_predict_trigger:
        if (name == '') or (description == '') or (skills == ''):
            st.success('Заполните все поля')
        else:
            cat = category_predict(name, skills, description, experience, schedule)
            st.success(cat)


    if salary_predict_trigger:

        def set_dummies_category(df, category):
            if category == 'Data Analysis':
                df['Data Analysis'] = 1
            elif category == 'Data Science':
                df['Data Science'] = 1
            elif category == 'Data Engineer':
                df['Data Engineer'] = 1
            else:
                df['Python Develop'] = 1

            return df

        def set_label_encoding_position_level(df, name = name):
            for word in name.split(' '):
                if word in ['Руководитель', 'Lead', 'Главный', 'Директор',
                            'Старший', 'Ведущий', 'Head', 'Senior',
                            'руководитель', 'lead', 'главный', 'директор',
                            'старший', 'ведущий', 'head', 'senior']:
                    df['Position_level'] = 3
                    return df
                elif word in ['junior', 'Junior', 'Стажер', 'Ассистент',
                              'Младший', 'Помощник', 'стажер', 'ассистент',
                              'младший', 'помощник']:
                    df['Position_level'] = 1
                    return df
                else:
                    df['Position_level'] = 2
                    return df

        df_sample_regr = pd.DataFrame([[0] * 7], columns=['Experience', 'Schedule', 'Position_level',
                                                          'Data Analysis', 'Data Engineer',
                                                          'Data Science', 'Python Develop'])
        if not cat:
            cat = category_predict(name, skills, description, experience, schedule,
                                   model_loaded=model_loaded,
                                   vectorize_columns_loaded=vectorize_columns_loaded)
        df_sample_regr = set_dummies_category(df_sample_regr, cat)
        df_sample_regr = set_label_encoding_position_level(df_sample_regr)
        df_sample_regr['Schedule'] = schedule
        df_sample_regr['Schedule'] = df_sample_regr['Schedule'].map({'Сменный график': 1,
                                                           'Гибкий график': 2,
                                                           'Удаленная работа': 3,
                                                           'Полный день': 3})
        df_sample_regr['Experience'] = experience
        df_sample_regr['Experience'] = df_sample_regr['Experience'].map({'Нет опыта': 1,
                                                 'От 1 года до 3 лет': 2,
                                                 'От 3 до 6 лет': 3,
                                                 'Более 6 лет': 4})


        df_for_predict_regr = pd.concat([pd.DataFrame([[skills, description]],
                                                 columns=['Skills', 'Description']), df_sample_regr], axis=1)

        data = load[1]
        model_loaded_regr = data["model_regr"]
        vectorize_columns_loaded_reg = data["vectorize_columns_regr"]

        tokenized_columns_regr = pd.DataFrame.sparse.from_spmatrix(vectorize_columns_loaded_reg.transform(df_for_predict_regr))

        df_for_predict_new_regr = pd.concat([tokenized_columns_regr, df_for_predict_regr.iloc[:, 2:].reset_index(drop=True)], axis=1)

        y_pred_regr = model_loaded_regr.predict(df_for_predict_new_regr)

        if (name == '') or (description == '') or (skills == ''):
            st.success('Заполните все поля')
        else:
            st.success(round(y_pred_regr[0]))








