import pandas as pd
import numpy as np

def evalTime(a, b):
    return (((a - b) / 100).astype(int) * 60).astype(float) + (((a % 100 + 60) - (b % 100)) % 60)

uid_train = pd.read_csv('../data/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('../data/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_train = pd.read_csv('../data/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_train = pd.read_csv('../data/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

voice_test = pd.read_csv('../data/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_test = pd.read_csv('../data/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_test = pd.read_csv('../data/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('../data/uid_test_b.txt',index=None)

voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)

voice['end_time'] = voice['end_time'].astype(float)
voice['start_time'] = voice['start_time'].astype(float)
voice['voice_time'] = evalTime(voice['end_time'], voice['start_time'])

voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index()
voice_opp_head=voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index()
voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)
voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)
voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)
voice_opp_num_count = voice.groupby(['uid'])['opp_num'].count().add_prefix('voice_opp_num_count_').reset_index().fillna(0)
voice_opp_diff_head = voice.groupby(['uid','opp_head'])['uid'].count().unstack().add_prefix('voice_opp_diff_head_').reset_index().fillna(0)
voice_opp_diff_len = voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_diff_len_').reset_index().fillna(0)
voice_length = voice.groupby(['uid'])['voice_time'].agg(['std','max','min','median','mean','sum']).add_prefix('voice_length_').reset_index().fillna(0)
voice_opp_head_inout = voice.groupby(['uid','opp_head'])['in_out'].count().unstack().add_prefix('voice_opp_head_inout_').reset_index().fillna(0)


sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index()
sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index()
sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)
sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)
sms_opp_num_count = sms.groupby(['uid'])['opp_num'].count().add_prefix('sms_opp_num_count_').reset_index().fillna(0)
sms_opp_diff_head = sms.groupby(['uid','opp_head'])['uid'].count().unstack().add_prefix('sms_opp_diff_head_').reset_index().fillna(0)
sms_opp_diff_len = sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_diff_len_').reset_index().fillna(0)
sms_opp_head_inout = sms.groupby(['uid','opp_head'])['in_out'].count().unstack().add_prefix('sms_opp_head_inout_').reset_index().fillna(0)

wa_name = wa.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index()
visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_').reset_index()
#visit_type_cnt_sum = wa.groupby(['uid','wa_type'])['visit_cnt'].sum().unstack().add_prefix('visit_type_cnt_sum_').reset_index().fillna(0)
visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_').reset_index()
#visit_type_dura_sum = wa.groupby(['uid','wa_type'])['visit_dura'].sum().unstack().add_prefix('visit_type_dura_sum_').reset_index().fillna(0)
up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_').reset_index()
down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_').reset_index()
wa_type = wa.groupby(['uid','wa_type'])['uid'].count().unstack().add_prefix('wa_visit_type_').reset_index().fillna(0)
#wa_type_up_flow = wa.groupby(['uid','wa_type'])['up_flow'].sum().unstack().add_prefix('wa_type_up_flow_').reset_index().fillna(0)
#wa_type_down_flow = wa.groupby(['uid','wa_type'])['down_flow'].sum().unstack().add_prefix('wa_type_down_flow_').reset_index().fillna(0)

feature = [voice_opp_num,voice_opp_head,voice_opp_len,voice_call_type,voice_in_out,voice_opp_num_count,voice_opp_diff_head,voice_opp_diff_len,voice_length,voice_opp_head_inout,sms_opp_num,sms_opp_head,sms_opp_len,sms_in_out,
            sms_opp_num_count,sms_opp_diff_head,sms_opp_diff_len,sms_opp_head_inout,wa_name,visit_cnt,visit_dura,up_flow,down_flow,wa_type]

train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature,feat,how='left',on='uid')

test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid')

train_feature.to_csv('../data/train_featureV1.csv',index=None)
test_feature.to_csv('../data/test_featureV1.csv',index=None)

del voice_opp_num
del voice_opp_head
del voice_opp_len
del voice_call_type
del voice_in_out
del voice_opp_num_count
del voice_opp_diff_head
del voice_opp_diff_len
del voice_length
del voice_opp_head_inout
del sms_opp_num
del sms_opp_head
del sms_opp_len
del sms_in_out
del sms_opp_num_count
del sms_opp_diff_head
del sms_opp_diff_len
del sms_opp_head_inout
del wa_name
del visit_cnt
#del visit_type_cnt_sum
del visit_dura
#del visit_type_dura_sum
del up_flow
del down_flow
#del wa_type_up_flow
#del wa_type_down_flow
del wa_type