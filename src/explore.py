import pandas as pd, numpy as np
v=pd.read_csv('data/dataset11-30.csv')
print(v.dtypes); print(v.describe())
v['start_time']=pd.to_datetime(v.start_time)
print(v.start_time.min(), v.start_time.max())
print('users',v.user_id.nunique(),'channels',v.vsetv_id.nunique())
sub=pd.read_csv('data/submission.csv')
print('test users in log', sub.user_id.isin(v.user_id).sum(), len(sub))
v['wk']=v.start_time.dt.isocalendar().week
print(v.groupby('wk').size())
for f in ['11-20','21-30','31-42']:
    s=pd.read_csv(f'data/export_arh_{f}-final.csv')
    s['st']=pd.to_datetime(s.start_time,format='%d.%m.%Y %H:%M:%S')
    print(f, s.st.min(), s.st.max(), 'labelled',(s.tv_show_id!=0).mean(), 'chan overlap with log', s.channel_id.isin(v.vsetv_id.unique()).mean())
    print(s.tv_show_category.value_counts().head(10))
