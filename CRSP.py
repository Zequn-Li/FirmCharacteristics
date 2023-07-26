import pandas as pd
import numpy as np
import wrds


file_path = '/Users/zequnli/LocalData/'
start_date = '1963-07-01'
end_date = '2021-12-31'


db = wrds.Connection(wrds_username='zli61')
query = ("select a.permno, a.date, a.ret, a.shrout, a.prc, b.exchcd,c.dlstcd, c.dlret from crsp.msf as a left join crsp.msenames as b on a.permno=b.permno and b.namedt<=a.date and a.date<=b.nameendt left join crsp.msedelist as c on a.permno=c.permno and date_trunc('month', a.date) = date_trunc('month', c.dlstdt)where date>= '{0}' and date <= '{1}'").format(start_date, end_date)
crsp = db.raw_sql(query, date_cols=['date'])
crsp.head()

# set dlret to -0.35 if dlret is null and dlstcd is 500 or 520-584 and exchcd is 1 or 2
conditions = [
    (crsp['dlret'].isnull()) &
    ((crsp['dlstcd'] == 500) | (crsp['dlstcd'] >= 520) & 
    (crsp['dlstcd'] <= 584)) &
    ((crsp['exchcd'] == 1) | (crsp['exchcd'] == 2))
    ]
values = [-0.35]
crsp['dlret'] = np.select(conditions, values, crsp['dlret'])

# if dlret less than -1, set to -1
crsp['dlret'] = np.where(crsp['dlret'] < -1, -1, crsp['dlret'])

# if dlret is missing, set to 0
crsp['dlret'] = np.where(crsp['dlret'].isnull(), 0, crsp['dlret'])

# calculate return including delisting return
crsp['retadj'] = (1 + crsp['ret']) * (1 + crsp['dlret']) - 1

# if the return is missing and dlret is not 0, set retadj to dlret
crsp['retadj'] = np.where((crsp['ret'].isnull()) & (crsp['dlret'] != 0), crsp['dlret'], crsp['retadj'])

# set the ret in percentage 
crsp['retadj'] = crsp['retadj'] * 100

# calculate market equity
crsp['me'] = crsp['prc'].abs() * crsp['shrout'] / 1000

# set the date to yyyymm format
crsp['yyyymm'] = crsp['date'].dt.year * 100 + crsp['date'].dt.month

# select data that have me available and not 0, target available
crsp = crsp[crsp['me'] != 0]
crsp = crsp[crsp['me'].notna()]
crsp = crsp[crsp['retadj'].notna()]

crsp = crsp[['permno', 'yyyymm', 'retadj', 'me']]
# change the data type of permno and yyyymm to int
crsp['permno'] = crsp['permno'].astype(int)
crsp['yyyymm'] = crsp['yyyymm'].astype(int)


# merge crsp with fama-french risk free rate
ff = pd.read_csv(file_path + 'ff.csv', index_col=0)
rf_dict = ff['RF'].to_dict()
crsp['rf'] = crsp['yyyymm'].map(rf_dict)
crsp['exret'] = crsp['retadj'] - crsp['rf']


crsp.to_csv(file_path + 'crsp.csv', index=False)