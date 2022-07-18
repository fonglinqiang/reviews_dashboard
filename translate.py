from google.cloud import translate_v2 as translate
from google.oauth2 import service_account

def run_gtranslation(g_credential,df):
    # g_credential: google credential uploaded as dictionary
    # df: dataframe to be translated

    # google credential setup
    credentials = service_account.Credentials.from_service_account_info(g_credential) 
    translate_client = translate.Client(credentials=credentials)

    # translation start
    global failed
    failed = 0

    def translate_review(text):
        try:
            target = 'en'
            output = translate_client.translate(str(text),target)
            return output['translatedText']
            # return 'translatedText'
        except:
            global failed
            failed += 1
            return '~' + text

    print(f'number of records: {len(df)}')
    print(f'estimated cost: {df.Review.str.len().sum()/1000000*20:,}')
    df['translated'] = df.Review.apply(translate_review)
    print('translation completed')
    print(f'number of failed translation: {failed}')

    return df


