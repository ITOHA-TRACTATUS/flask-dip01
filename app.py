import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                             ExtraTreesRegressor, GradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb
import flask
import joblib
app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return '''
    <form method="post" action="/upload" enctype="multipart/form-data">
      <input type="file" name="file">
      <button>upload</button>
      <h4>５分少々お待ちください。自動的にファイルはダウンロードされます。</h4>
    </form>
'''

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in flask.request.files:
        return 'ファイル未指定'

    # https://tedboy.github.io/flask/generated/generated/werkzeug.FileStorage.html
    fs = flask.request.files['file']

    app.logger.info('file_name={}'.format(fs.filename))
    app.logger.info('content_type={} content_length={}, mimetype={}, mimetype_params={}'.format(
        fs.content_type, fs.content_length, fs.mimetype, fs.mimetype_params))

    fs.save('upload/test_x.csv')

    train_x = pd.read_csv('trained-model/train_x.csv', encoding="utf8", na_values=['missing'])
    train_y = pd.read_csv('trained-model/train_y.csv', encoding="utf8", na_values=['missing'])
    train_y = train_y.drop('お仕事No.', axis=1)
    data = pd.concat([train_x, train_y], axis=1)

    test = pd.read_csv('upload/test_x.csv', encoding="utf8", na_values=['missing'])

    data['動画コメント'] = data['動画コメント'].fillna('空欄')
    test['動画コメント'] = test['動画コメント'].fillna('空欄')

    for i in range(len(data)):
        movie_comment_num = len(str(data['動画コメント'].iloc[i]))
        data['動画コメント'].iloc[i] = movie_comment_num
    for i in range(len(test)):
        movie_comment_num = len(str(test['動画コメント'].iloc[i]))
        test['動画コメント'].iloc[i] = movie_comment_num

    data['休日休暇　備考'] = data['休日休暇　備考'].fillna('空欄')
    test['休日休暇　備考'] = test['休日休暇　備考'].fillna('空欄')

    for i in range(len(data)):
        holiday_remarks_num = len(str(data['休日休暇　備考'].iloc[i]))
        data['休日休暇　備考'].iloc[i] = holiday_remarks_num
    for i in range(len(test)):
        holiday_remarks_num = len(str(test['休日休暇　備考'].iloc[i]))
        test['休日休暇　備考'].iloc[i] = holiday_remarks_num

    data['（派遣）応募後の流れ'] = data['（派遣）応募後の流れ'].fillna('空欄')
    test['（派遣）応募後の流れ'] = test['（派遣）応募後の流れ'].fillna('空欄')

    for i in range(len(data)):
        after_application_num = len(str(data['（派遣）応募後の流れ'].iloc[i]))
        data['（派遣）応募後の流れ'].iloc[i] = after_application_num
    for i in range(len(test)):
        after_application_num = len(str(test['（派遣）応募後の流れ'].iloc[i]))
        test['（派遣）応募後の流れ'].iloc[i] = after_application_num

    data['応募資格'] = data['応募資格'].fillna('空欄')
    test['応募資格'] = test['応募資格'].fillna('空欄')

    data['inexperience_person'] = data["応募資格"].apply(lambda x: 0 if '未経験' in x else 1)
    data['university graduate'] = data["応募資格"].apply(lambda x: 0 if '大卒' in x else 1)
    data['qualification'] = data["応募資格"].apply(lambda x: 0 if '級' in x else 1)

    test['inexperience_person'] = test["応募資格"].apply(lambda x: 0 if '未経験' in x else 1)
    test['university graduate'] = test["応募資格"].apply(lambda x: 0 if '大卒' in x else 1)
    test['qualification'] = test["応募資格"].apply(lambda x: 0 if '級' in x else 1)

    for i in range(len(data)):
        qualification_num = len(str(data['応募資格'].iloc[i]))
        data['応募資格'].iloc[i] = qualification_num
    for i in range(len(test)):
        qualification_num = len(str(test['応募資格'].iloc[i]))
        test['応募資格'].iloc[i] = qualification_num

    data['派遣会社のうれしい特典'] = data['派遣会社のうれしい特典'].fillna('空欄')
    test['派遣会社のうれしい特典'] = test['派遣会社のうれしい特典'].fillna('空欄')

    for i in range(len(data)):
        special_gift_num = len(str(data['派遣会社のうれしい特典'].iloc[i]))
        data['派遣会社のうれしい特典'].iloc[i] = special_gift_num
    for i in range(len(test)):
        special_gift_num = len(str(test['派遣会社のうれしい特典'].iloc[i]))
        test['派遣会社のうれしい特典'].iloc[i] = special_gift_num

    data['お仕事のポイント（仕事PR）'] = data['お仕事のポイント（仕事PR）'].fillna('空欄')
    test['お仕事のポイント（仕事PR）'] = test['お仕事のポイント（仕事PR）'].fillna('空欄')

    for i in range(len(data)):
        public_relations_num = len(str(data['お仕事のポイント（仕事PR）'].iloc[i]))
        data['お仕事のポイント（仕事PR）'].iloc[i] = public_relations_num
    for i in range(len(test)):
        public_relations_num = len(str(test['お仕事のポイント（仕事PR）'].iloc[i]))
        test['お仕事のポイント（仕事PR）'].iloc[i] = public_relations_num

    data['（派遣先）職場の雰囲気'] = data['（派遣先）職場の雰囲気'].fillna('空欄')
    test['（派遣先）職場の雰囲気'] = test['（派遣先）職場の雰囲気'].fillna('空欄')

    for i in range(len(data)):
        workplace_environment_num = len(str(data['（派遣先）職場の雰囲気'].iloc[i]))
        data['（派遣先）職場の雰囲気'].iloc[i] = workplace_environment_num
    for i in range(len(test)):
        workplace_environment = len(str(test['（派遣先）職場の雰囲気'].iloc[i]))
        test['（派遣先）職場の雰囲気'].iloc[i] = workplace_environment_num

    data['給与/交通費　備考'] = data['給与/交通費　備考'].fillna('空欄')
    test['給与/交通費　備考'] = test['給与/交通費　備考'].fillna('空欄')

    for i in range(len(data)):
        expenses_remarks_num = len(str(data['給与/交通費　備考'].iloc[i]))
        data['給与/交通費　備考'].iloc[i] = expenses_remarks_num
    for i in range(len(test)):
        expenses_remarks = len(str(test['給与/交通費　備考'].iloc[i]))
        test['給与/交通費　備考'].iloc[i] = expenses_remarks_num

    data['期間･時間　備考'] = data['期間･時間　備考'].fillna('空欄')
    test['期間･時間　備考'] = test['期間･時間　備考'].fillna('空欄')

    for i in range(len(data)):
        time_remarks_num = len(str(data['期間･時間　備考'].iloc[i]))
        data['期間･時間　備考'].iloc[i] = time_remarks_num
    for i in range(len(test)):
        time_remarks = len(str(test['期間･時間　備考'].iloc[i]))
        test['期間･時間　備考'].iloc[i] = time_remarks_num
        
    data['（派遣先）配属先部署　男女比　男'] = data['（派遣先）配属先部署　男女比　男'].fillna(11)
    data['（派遣先）配属先部署　男女比　女'] = data['（派遣先）配属先部署　男女比　女'].fillna(11)
    data['（派遣先）配属先部署　平均年齢'] = data['（派遣先）配属先部署　平均年齢'].fillna(data['（派遣先）配属先部署　平均年齢'].mean())

    test['（派遣先）配属先部署　男女比　男'] = test['（派遣先）配属先部署　男女比　男'].fillna(11)
    test['（派遣先）配属先部署　男女比　女'] = test['（派遣先）配属先部署　男女比　女'].fillna(11)
    test['（派遣先）配属先部署　平均年齢'] = test['（派遣先）配属先部署　平均年齢'].fillna(test['（派遣先）配属先部署　平均年齢'].mean())

    data = data.drop(['勤務地　最寄駅3（駅名）', 'オープニングスタッフ', '未使用.20', 'メモ', '（紹介予定）年収・給与例',
                    'WEB面接OK', '応募先　備考', '応募拠点', '固定残業制 残業代 下限', '未使用.12', 'シニア（60〜）歓迎',
                    '（派遣先）概要　従業員数', '未使用.14', '未使用.5', '未使用.3', '勤務地　最寄駅2（駅名）', '募集形態',
                    'ベンチャー企業', '未使用.13', '（派遣以外）応募後の流れ', '経験必須', '（派遣先）概要　勤務先名（フリガナ）',
                    '期間・時間　勤務開始日', 'フラグオプション選択', '勤務地　最寄駅3（沿線名）', '勤務地　最寄駅2（沿線名）',
                    'フリー項目　内容', 'ブロックコード3', '未使用.16', 'バイク・自転車通勤OK', '未使用.9', '動画タイトル',
                    '勤務地　最寄駅1（沿線名）', '週払い', '外国人活躍中・留学生歓迎', '人材紹介', '（紹介予定）雇用形態備考',
                    'これまでの採用者例', '仕事写真（下）　写真3　コメント', '（派遣先）概要　勤務先名（漢字）', '日払い',
                    'フリー項目　タイトル', '未使用.11', '拠点番号',
                    '給与　経験者給与下限', '未使用.8', '（派遣先）勤務先写真コメント', '無期雇用派遣', '応募先　所在地　ブロックコード',
                    'Wワーク・副業可能', '待遇・福利厚生', '主婦(ママ)・主夫歓迎', '未使用.18', '給与/交通費　給与上限', '未使用.15',
                    '（紹介予定）入社後の雇用形態', '少人数の職場', '固定残業制 残業代に充当する労働時間数 下限', '未使用.4',
                    '勤務地　周辺情報', '未使用.10', '応募先　所在地　都道府県', '学生歓迎', '先輩からのメッセージ',
                    '未使用.1', '給与　経験者給与上限', '17時以降出社OK', 'エルダー（50〜）活躍中', '（紹介予定）待遇・福利厚生',
                    '未使用.7', '仕事写真（下）　写真2　コメント', '電話応対なし', 'ネットワーク関連のスキルを活かす',
                    'プログラム関連のスキルを活かす', '未使用.21', '仕事写真（下）　写真1　ファイル名', '未使用.2',
                    '仕事写真（下）　写真1　コメント', 'ブランクOK', '（派遣先）概要　事業内容', '未使用.17', '未使用.22',
                    '勤務地　最寄駅2（分）', '（紹介予定）休日休暇', 'ブロックコード2', '（紹介予定）入社時期', '（派遣先）配属先部署　人数',
                    '応募先　所在地　市区町村', '未使用', '寮・社宅あり', '応募先　最寄駅（沿線名）', '勤務地　最寄駅2（駅からの交通手段）',
                    '仕事写真（下）　写真2　ファイル名', '応募先　最寄駅（駅名）', '勤務地　最寄駅3（分）',
                    '固定残業制 残業代に充当する労働時間数 上限', '（派遣先）勤務先写真ファイル名',
                    '固定残業制 残業代 上限', '未使用.6', '仕事写真（下）　写真3　ファイル名', '勤務地　最寄駅3（駅からの交通手段）',
                    'ブロックコード1', '未使用.19', 'WEB関連のスキルを活かす', '勤務地　最寄駅1（駅からの交通手段）', '応募先　名称',
                    '（派遣先）配属先部署', '勤務地　最寄駅1（分）', '掲載期間　開始日', '期間・時間　勤務時間',
                    '勤務地　備考', 'お仕事名', '仕事内容', '掲載期間　終了日', '動画ファイル名', '勤務地　最寄駅1（駅名）'], axis=1)

    test = test.drop(['勤務地　最寄駅3（駅名）', 'オープニングスタッフ', '未使用.20', 'メモ', '（紹介予定）年収・給与例',
                    'WEB面接OK', '応募先　備考', '応募拠点', '固定残業制 残業代 下限', '未使用.12', 'シニア（60〜）歓迎',
                    '（派遣先）概要　従業員数', '未使用.14', '未使用.5', '未使用.3', '勤務地　最寄駅2（駅名）', '募集形態',
                    'ベンチャー企業', '未使用.13', '（派遣以外）応募後の流れ', '経験必須', '（派遣先）概要　勤務先名（フリガナ）',
                    '期間・時間　勤務開始日', 'フラグオプション選択', '勤務地　最寄駅3（沿線名）', '勤務地　最寄駅2（沿線名）',
                    'フリー項目　内容', 'ブロックコード3', '未使用.16', 'バイク・自転車通勤OK', '未使用.9', '動画タイトル',
                    '勤務地　最寄駅1（沿線名）', '週払い', '外国人活躍中・留学生歓迎', '人材紹介', '（紹介予定）雇用形態備考',
                    'これまでの採用者例', '仕事写真（下）　写真3　コメント', '（派遣先）概要　勤務先名（漢字）', '日払い',
                    'フリー項目　タイトル', '未使用.11', '拠点番号',
                    '給与　経験者給与下限', '未使用.8', '（派遣先）勤務先写真コメント', '無期雇用派遣', '応募先　所在地　ブロックコード',
                    'Wワーク・副業可能', '待遇・福利厚生', '主婦(ママ)・主夫歓迎', '未使用.18', '給与/交通費　給与上限', '未使用.15',
                    '（紹介予定）入社後の雇用形態', '少人数の職場', '固定残業制 残業代に充当する労働時間数 下限', '未使用.4',
                    '勤務地　周辺情報', '未使用.10', '応募先　所在地　都道府県', '学生歓迎', '先輩からのメッセージ',
                    '未使用.1', '給与　経験者給与上限', '17時以降出社OK', 'エルダー（50〜）活躍中', '（紹介予定）待遇・福利厚生',
                    '未使用.7', '仕事写真（下）　写真2　コメント', '電話応対なし', 'ネットワーク関連のスキルを活かす',
                    'プログラム関連のスキルを活かす', '未使用.21', '仕事写真（下）　写真1　ファイル名', '未使用.2',
                    '仕事写真（下）　写真1　コメント', 'ブランクOK', '（派遣先）概要　事業内容', '未使用.17', '未使用.22',
                    '勤務地　最寄駅2（分）', '（紹介予定）休日休暇', 'ブロックコード2', '（紹介予定）入社時期', '（派遣先）配属先部署　人数',
                    '応募先　所在地　市区町村', '未使用', '寮・社宅あり', '応募先　最寄駅（沿線名）', '勤務地　最寄駅2（駅からの交通手段）',
                    '仕事写真（下）　写真2　ファイル名', '応募先　最寄駅（駅名）', '勤務地　最寄駅3（分）',
                    '固定残業制 残業代に充当する労働時間数 上限', '（派遣先）勤務先写真ファイル名',
                    '固定残業制 残業代 上限', '未使用.6', '仕事写真（下）　写真3　ファイル名', '勤務地　最寄駅3（駅からの交通手段）',
                    'ブロックコード1', '未使用.19', 'WEB関連のスキルを活かす', '勤務地　最寄駅1（駅からの交通手段）', '応募先　名称',
                    '（派遣先）配属先部署', '勤務地　最寄駅1（分）', '掲載期間　開始日', '期間・時間　勤務時間',
                    '勤務地　備考', 'お仕事名', '仕事内容', '掲載期間　終了日', '動画ファイル名', '勤務地　最寄駅1（駅名）'], axis=1)

    
    features_to_scale = data.drop(['お仕事No.'], axis=1).columns
    scaler = pp.StandardScaler(copy=True)
    data.loc[:, features_to_scale] = scaler.fit_transform(data[features_to_scale])
    
    features_to_scale = test.drop(['お仕事No.'], axis=1).columns
    scaler = pp.StandardScaler(copy=True)
    test.loc[:, features_to_scale] = scaler.fit_transform(test[features_to_scale])


    ntrain = data.shape[0]
    ntest = test.shape[0]
    SEED = 0
    NFOLDS = 5
    kf = KFold(n_splits=NFOLDS, random_state=SEED)

    class SklearnHelper(object):
        def __init__(self, clf, seed=0, params=None):
            params['random_state'] = seed
            self.clf = clf(**params)

        def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)

        def predict(self, x):
            return self.clf.predict(x)

        def fit(self,x,y):
            return self.clf.fit(x,y)

        def feature_importances(self,x,y):
            print(self.clf.fit(x,y).feature_importances_)

            
    def get_oof(clf, x_train, y_train, x_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf.split(x_train,y_train)):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    # Random Forest
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        'warm_start': True, 
        #'max_features': 0.2,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features' : 'sqrt',
        'verbose': 0
    }

    # Extra Trees
    et_params = {
        'n_jobs': -1,
        'n_estimators':500,
        #'max_features': 0.5,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    # AdaBoost
    ada_params = {
        'n_estimators': 500,
        'learning_rate' : 0.75
    }

    # Gradient Boosting
    gb_params = {
        'n_estimators': 500,
        #'max_features': 0.2,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    gb_params1 = {'learning_rate' : 0.1,
                'max_depth' : 2,
                'subsample' : 0.5
                }

    gb_params2 = {'learning_rate' : 0.3,
                'max_depth' : 3,
                'subsample' : 0.5
                }
                
    gb_params3 = {'learning_rate' : 0.3,
                'max_depth' : 5,
                'subsample' : 1.0
                }
                
    gb_params4 = {'learning_rate' : 0.5,
                'max_depth' : 10,
                'subsample' : 1.0
                }

    rf = SklearnHelper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
    et = SklearnHelper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
    ada = SklearnHelper(clf=AdaBoostRegressor, seed=SEED, params=ada_params)
    gb = SklearnHelper(clf=GradientBoostingRegressor, seed=SEED, params=gb_params)
    xg1 = SklearnHelper(clf=xgb.XGBRegressor, seed=SEED, params=gb_params1)
    xg2 = SklearnHelper(clf=xgb.XGBRegressor, seed=SEED, params=gb_params2)
    xg3 = SklearnHelper(clf=xgb.XGBRegressor, seed=SEED, params=gb_params3)
    xg4 = SklearnHelper(clf=xgb.XGBRegressor, seed=SEED, params=gb_params4)

    y_train = data['応募数 合計'].ravel()
    train = data.drop(['応募数 合計'], axis=1)
    x_train = train.values
    x_test = test.values

    et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
    rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test)
    ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)
    gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test)
    xg1_oof_train, xg1_oof_test = get_oof(xg1,x_train,y_train,x_test)
    xg2_oof_train, xg2_oof_test = get_oof(xg2,x_train,y_train,x_test)
    xg3_oof_train, xg3_oof_test = get_oof(xg3,x_train,y_train,x_test)
    xg4_oof_train, xg4_oof_test = get_oof(xg4,x_train,y_train,x_test)

    base_predictions_train = pd.DataFrame( {
        'RandomForest': rf_oof_train.ravel(),
        'ExtraTrees': et_oof_train.ravel(),
        'AdaBoost': ada_oof_train.ravel(),
        'GradientBoost': gb_oof_train.ravel(),
        'XGBoost1': xg1_oof_train.ravel(),
        'XGBoost2': xg2_oof_train.ravel(),
        'XGBoost3': xg3_oof_train.ravel(),
        'XGBoost4': xg4_oof_train.ravel()
    })
    
    x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train,
                        xg1_oof_train, xg2_oof_train, xg3_oof_train, xg4_oof_train), axis=1)
    x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test,
                        xg1_oof_test, xg2_oof_test, xg3_oof_test, xg4_oof_test), axis=1)

    ridge = Ridge()
    model = ridge.fit(x_train, y_train)

    predictions = model.predict(x_test)

    job_number_test = test['お仕事No.'].copy()
    predictions_pd = pd.Series(data = predictions, name='応募数 合計')
    submit = pd.concat([job_number_test, predictions_pd], axis=1)
    submit.to_csv("export/test_y.csv", index=False, encoding='utf-8')

    downloadFileName = 'test_y.csv'
    downloadFile = 'export/test_y.csv' 
    return flask.send_file(downloadFile, as_attachment = True, attachment_filename = downloadFileName)


if __name__ == "__main__":

    app.run(debug=False)