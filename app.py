import os
from io import TextIOWrapper
from flask import Flask,request,jsonify,render_template,Response, request, redirect, url_for,session,send_from_directory
import secrets
from functools import wraps
import psycopg2
import pandas as pd
import json
from model import run_model, read_output
from clean import clean_cache
from config import HOST,DATABASE,USER,PASSWORD,PORT,TABLE,CLEAN
from translate import run_gtranslation
from calendar import monthrange
from datetime import datetime


app = Flask(__name__)
app.secret_key = 'your-secret-key'

results_files = [f for f in os.listdir('./model_output') if f[:4]=='app_']
results_files.sort(reverse=True)
results_files = [f[4:11] for f in results_files]
model_output = read_output(results_files[0])


def api_key_required(foo):
    @wraps(foo)
    def wrap(*args, **kwargs):
        if 'user' in session:
            return foo(*args, **kwargs)
        else:
            print('out')
            return redirect(url_for('home'))
    return wrap


@app.route('/')
def home():
    return render_template('home.html')


# Step 1: Get new records
@app.route('/new', defaults={'download':None}, methods=['GET', 'POST'])
@app.route('/new/<download>')
@api_key_required
def new(download):
    if request.method == 'POST':
        print('Uploading files ...')
        file_old = request.files['file_old']
        file_new = request.files['file_new']
        print('Reading uploaded files ...')
        df_old = pd.read_excel(file_old)
        df_new = pd.read_excel(file_new)

        print('find new reviews ...')
        df = pd.concat([df_old,df_new]).drop_duplicates(keep=False)
        df = df[df['Review'].notna()]
        df['Start Date'] = df['Start Date'].dt.strftime('%Y-%m-%d')
        df['End Date'] = df['End Date'].dt.strftime('%Y-%m-%d')
        df.to_csv('data/Reviews_new.csv',index=False)
        print('Reviews_new.csv has been updated')

        return render_template('new_output.html',num_new=len(df),est_cost=f'{df.Review.str.len().sum()/1000000*20:,}')
    elif download == 'download':
        try:
            return send_from_directory(directory='/Users/sam/Developer/machineLearning/users_review_ML/users_review_sentiment_deploy/data', filename='Reviews_new.csv', as_attachment=True, attachment_filename='Reviews_new.csv')
        except:
            return "Reviews_new.csv does not exist."

    return render_template('new.html')


# Step 2: Translate Reviews using Google Translate API
@app.route('/translate', methods=['GET','POST'])
@api_key_required
def translate():
    if request.method == 'POST':
        g_credential_json = request.files['g_credential']
        g_credential = json.load(g_credential_json)
        # print(g_credential)
        pre_translated = request.files['pre_translated']
        df_pre_translated = pd.read_csv(pre_translated)

        # directory is different in docker!!!
        df_post_translated = run_gtranslation(g_credential=g_credential,df=df_pre_translated)
        df_post_translated.to_csv('data/Reviews_new_gtranslated.csv',index=False)
        return send_from_directory(directory='/Users/sam/Developer/machineLearning/users_review_ML/users_review_sentiment_deploy/data', filename='Reviews_new_gtranslated.csv', as_attachment=True, attachment_filename='Reviews_new_gtranslated.csv')
    
    return render_template('translate.html')


# Step 3: Upload translated text into database
@app.route('/upload', methods=['GET', 'POST'])
@api_key_required
def upload():
    if request.method == 'POST':
        # todo - update credentials accoordingly
        con = psycopg2.connect(host = HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)

        cur = con.cursor()
        count = 0

        csv_file = request.files['file']
        csv_file = TextIOWrapper(csv_file, encoding='utf-8')
        df = pd.read_csv(csv_file)
        df.info()

        # todo - Start Date, End Date date format change accordingly, changed to YYYY-MM-DD , since exported csv has this format
        for i,row in df.iterrows():
            cur.execute(f"""
            INSERT INTO 
            {TABLE}(
                "ID", 
                "Review", 
                "Rating", 
                "Start Date", 
                "End Date", 
                "Review Product Version", 
                "Review Product Version Name", 
                "Country/Region/Language", 
                "Country/Region/Language Name", 
                "Platform", 
                "translated")
            VALUES 
                ('{str(row['ID'])}', 
                '{str(row['Review']).replace("'", "''", 99)}', 
                '{row['Rating']}', 
                TO_DATE('{row['Start Date']}','YYYY-MM-DD'), 
                TO_DATE('{row['End Date']}','YYYY-MM-DD'), 
                '{str(row['Review Product Version'])}', 
                '{str(row['Review Product Version Name'])}', 
                '{row['Country/Region/Language']}', 
                '{row['Country/Region/Language Name'].replace("'", "''", 99)}', 
                '{row['Platform']}', 
                '{str(row['translated']).replace("'", "''", 99)}');
            """)
            count+=1

        con.commit()
        print(count, f"Record inserted successfully into {TABLE}.")
        con.close()
        # global model_output
        # model_output = run_model()
        return redirect(url_for('db'))
    return render_template('upload.html')


# Step 4: Check number of records in database
@app.route('/db')
@api_key_required
def db():
    #establishing the connection
    conn = psycopg2.connect(host = HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)

    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    # Check DB connection/version
    cursor.execute("select version()")
    version = cursor.fetchone()
    print("Connection established to: ",version[0])

    # Get number of records in Reviews
    cursor.execute(f"select count(*) from {TABLE}")
    records = cursor.fetchone()
    print(f"Number of records in {TABLE}: ",records[0])

    # Get number of records in Clean
    cursor.execute(f"select count(*) from {CLEAN}")
    records2 = cursor.fetchone()
    print(f"Number of records in {CLEAN}: ",records2[0])

    #Closing the connection
    conn.close()
    return render_template('database.html', version=version[0], review_table=TABLE, records=records[0], clean_table=CLEAN, records2=records2[0])


# Step 5: Clean and Cache Reviews
@app.route('/clean')
@api_key_required
def clean():
    cleaned_records,review_records = clean_cache()
    return render_template('clean.html', clean_table=CLEAN,cleaned_records=cleaned_records,review_table=TABLE,review_records=review_records)


# Step 6: Run model and update cached results
@app.route('/update')
@api_key_required
def update():
    print('Fetch data from DB')
    con = psycopg2.connect(host = HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)
    sql = f"""
            select * from {TABLE}
            inner join {CLEAN}
            on {TABLE}."ReviewID" = {CLEAN}."ReviewID"
            """
    # df = pd.read_sql_query(f'select * from {TABLE}',con=con,index_col='ReviewID')
    df = pd.read_sql_query(sql,con=con)
    con.close()

    df.columns=['ReviewID', 'ID', 'Rating', 'Start Date', 'End Date',
       'Review Product Version', 'Review Product Version Name',
       'Country/Region/Language', 'Country/Region/Language Name', 'Platform',
       'Review', 'translated', 'ReviewID2', 'review_readable',
       'review_cleaned']
    df.drop(columns=['ReviewID2'],inplace=True)
    df.set_index('ReviewID')

    def grouping(col):
        return f'{col.year}-{col.month:02d}'

    df['Start Date'] = pd.to_datetime(df['Start Date'],format='%Y/%m/%d')
    df['grouping'] = df['Start Date'].apply(grouping)
    update_list = set(df['grouping'].unique().tolist()).symmetric_difference(set(results_files))
    update_list = [i for i in update_list if int(i[:4])>2011]

    for group in update_list:
        print(f'----- {group} -----')

        df_app = df[df['grouping'] == group]
        print(f'Assigning {group} to df_app with {len(df_app)} records')
        df_app.drop(['grouping'],axis=1,inplace=True)
        output_app = run_model(df_app)

        with open(f'model_output/app_{group}.json', 'w') as outfile:
            json.dump(output_app, outfile)
            print(f'model_output/app_{group}.json has been updated')

        df_ios = df_app[df_app['Platform'] == 'iOS']
        print(f'Assigning iOS to df_ios with {len(df_ios)} records')
        output_ios = run_model(df_ios)

        with open(f'model_output/iOS_{group}.json', 'w') as outfile:
            json.dump(output_ios, outfile)
            print(f'model_output/iOS_{group}.json has been updated')

        df_Android = df_app[df_app['Platform'] == 'Android']
        print(f'Assigning Android to df_Android with {len(df_Android)} records')
        output_Android = run_model(df_Android)

        with open(f'model_output/Android_{group}.json', 'w') as outfile:
            json.dump(output_Android, outfile)
            print(f'model_output/Android_{group}.json has been updated')

    return redirect(url_for('report'))


# Step 7: View report
@app.route('/report',defaults={'key':None},methods=['GET','POST'])
@app.route('/report/<key>')
def report(key):
    global results_files
    global model_output
    results_files = [f for f in os.listdir('./model_output') if f[:4]=='app_']
    results_files.sort(reverse=True)
    results_files = [f[4:11] for f in results_files]

    con = psycopg2.connect(host = HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)
    cur = con.cursor()
    sql = f'select distinct "Start Date" from {TABLE} order by "Start Date" desc'
    cur.execute(sql)
    date_range=cur.fetchall()
    con.close()
    date_range = [d[0].strftime("%Y-%m-%d") for d in date_range]
    if 'user' in session:
        if request.method == 'POST':
            # get filtered options
            filter_date_start,filter_date_end,wholemonth = check_wholemonth(request.form.get('filter_date_start'),request.form.get('filter_date_end'))
            filter_os = request.form.get('filter_os')

            if wholemonth == True:
                # get from cache model_output
                filter_date = filter_date_start[:7]
                return render_template('review_dashboard.html',model_output=read_output(filter_date,filter_os),date_range=date_range,results_files=results_files)
            else:
                # run model using start and end dates
                con = psycopg2.connect(host = HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)
                sql = f"""
                select * from {TABLE}
                inner join {CLEAN}
                on {TABLE}."ReviewID" = {CLEAN}."ReviewID"
                where "Start Date" >= '{filter_date_start}' and "Start Date" <= '{filter_date_end}'
                """
                df = pd.read_sql_query(sql,con=con)
                con.close()
                df.columns=['ReviewID', 'ID', 'Rating', 'Start Date', 'End Date',
                    'Review Product Version', 'Review Product Version Name',
                    'Country/Region/Language', 'Country/Region/Language Name', 'Platform',
                    'Review', 'translated', 'ReviewID2', 'review_readable',
                    'review_cleaned']
                df.drop(columns=['ReviewID2'],inplace=True)
                df.set_index('ReviewID',inplace=True)
                model_output = run_model(df)
                return render_template('review_dashboard.html',model_output=model_output,date_range=date_range,results_files=results_files)
        else:
            return render_template('review_dashboard.html',model_output=model_output,date_range=date_range,results_files=results_files)
            
    else:
        if app.secret_key == key: 
            session['user'] = secrets.token_urlsafe(16)
            return render_template('review_dashboard.html',model_output=model_output,date_range=date_range,results_files=results_files)
        else:
            return redirect(url_for('home'))


# # testing
# @app.route('/test',defaults={'key':None},methods=['GET','POST'])
# @api_key_required
# def test(key):
#     global results_files
#     global model_output
#     results_files = [f for f in os.listdir('./model_output') if f[:4]=='app_']
#     results_files.sort(reverse=True)
#     results_files = [f[4:11] for f in results_files]

#     con = psycopg2.connect(host = HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)
#     cur = con.cursor()
#     sql = f'select distinct "Start Date" from {TABLE} order by "Start Date" desc'
#     cur.execute(sql)
#     date_range=cur.fetchall()
#     con.close()
#     date_range = [d[0].strftime("%Y-%m-%d") for d in date_range]

#     if request.method == 'POST':
#         # get filtered options
#         filter_date_start,filter_date_end,wholemonth = check_wholemonth(request.form.get('filter_date_start'),request.form.get('filter_date_end'))
#         filter_os = request.form.get('filter_os')

#         if wholemonth == True:
#             # get from cache model_output
#             filter_date = filter_date_start[:7]
#             return render_template('review_dashboard.html',model_output=read_output(filter_date,filter_os),date_range=date_range,results_files=results_files)
#         else:
#             # run model using start and end dates
#             con = psycopg2.connect(host = HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)
#             sql = f"""
#             select * from {TABLE}
#             inner join {CLEAN}
#             on {TABLE}."ReviewID" = {CLEAN}."ReviewID"
#             where "Start Date" >= '{filter_date_start}' and "Start Date" <= '{filter_date_end}'
#             """
#             df = pd.read_sql_query(sql,con=con)
#             con.close()
#             df.columns=['ReviewID', 'ID', 'Rating', 'Start Date', 'End Date',
#                 'Review Product Version', 'Review Product Version Name',
#                 'Country/Region/Language', 'Country/Region/Language Name', 'Platform',
#                 'Review', 'translated', 'ReviewID2', 'review_readable',
#                 'review_cleaned']
#             df.drop(columns=['ReviewID2'],inplace=True)
#             df.set_index('ReviewID',inplace=True)
#             model_output = run_model(df)
#             return render_template('review_dashboard.html',model_output=model_output,date_range=date_range,results_files=results_files)
            
#     else:
#         return render_template('review_dashboard.html',model_output=model_output,date_range=date_range,results_files=results_files)


@app.route('/logout')
def logout():
    session.pop('user',None)
    return redirect(url_for('home'))


# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=5000)


def check_wholemonth(date1,date2):
    date1_date = datetime.strptime(date1,'%Y-%m-%d')
    date2_date = datetime.strptime(date2,'%Y-%m-%d')
    if date1_date < date2_date:
        date_start = date1
        date_end = date2
    else:
        date_start = date2
        date_end = date1
    date_diff = date2_date - date1_date
    date_diff = abs(date_diff.days)+1
    if date1[:7] == date2[:7]:
        if int(monthrange(date1_date.year,date1_date.month)[1]) == date_diff:
            return date_start,date_end,True
    return date_start,date_end,False