import pandas as pd
import re
import numpy as np
def load_train_df(paper_dir="paper_machine_data-788605.csv",train_dir="/data/work/ccf/train-974578.csv"):
    df_paper=pd.read_csv(paper_dir,index_col=0)

    df_train=pd.read_csv(train_dir,index_col=0)
    #debug
    df_paper=df_paper[:50]
    df_train=df_train[:50]    
    # nan value
    
    df_paper.dropna(axis=0,inplace=True)
    df_train.dropna(axis=0,inplace=True)
    
    df_paper = process_paper(df_paper)
    df_train = process_train(df_train)
    
    df_train = merge_dfs(df_paper,df_train)
    
def merge_dfs(df_paper:pd.DataFrame,df_train:pd.DataFrame)->pd.DataFrame:
    timediff = pd.Timedelta('1 days')

    # define model
    from sklearn.linear_model import LogisticRegression # 预测器
    model = LogisticRegression()
    
    # for idx,row_train in df_train.iterrows():
    #     df_paper_train_temp=df_paper
    #     # df_paper_train_temp=df_paper_train_temp[(row_train['end_time']-df_paper_train_temp['time'])<timediff] # 限制end_time和加工间隔
    #     df_paper_train_temp=df_paper_train_temp[df_paper_train_temp['time']<=row_train['end_time']]

    #     if (df_paper_train_temp.shape[0]!=0):
    #         df_paper_train_temp= df_paper_train_temp.drop(['time'],axis=1)
    #         df_paper_train_temp.dropna(inplace=True)
    #         X = df_paper_train_temp.values
    #         Y = np.full(df_paper_train_temp.shape[0],row_train['paper_tension_vertically_average'])
    #         model.fit(X,Y)
    #     del df_paper_train_temp
        
    for idx,row_paper in df_paper.iterrows():
        df_train_temp=df_train
        # df_paper_train_temp=df_paper_train_temp[(row_train['end_time']-df_paper_train_temp['time'])<timediff] # 限制end_time和加工间隔
        df_train_temp=df_train_temp[df_train_temp['end_time']>=row_paper['time']]

        if (df_train_temp.shape[0]!=0):
            df_train_temp.dropna(inplace=True)
            Y = np.array(df_train_temp['paper_tension_vertically_average'])
            df_train_temp= df_train_temp.drop(['end_time','paper_tension_vertically_average'],axis=1)
            X = df_train_temp.values
            
            row_paper=row_paper.drop('time').values
            expand_X=np.broadcast_to(row_paper,(X.shape[0],row_paper.shape[0]))
            X = np.concatenate((X,expand_X),axis=1)
            assert X.shape[0]== Y.shape[0]
            model.fit(X,Y)
        del df_train_temp
        
    return model
    
def process_train(df:pd.DataFrame)->pd.DataFrame:
    
    def formula_pre(row):
        row_str=row["formula"]
        pattern=re.compile(r"(\d+\.\d*%)")
        matchObj = pattern.findall(row_str)
        assert len(matchObj)==2,"formula lenth error"
        matchObj = [float(num.strip('%')) for num in matchObj]
        return matchObj
        
    formula_split=df.apply(formula_pre,axis=1)# process formula
    formula_split=np.array(formula_split.to_list())
    df['formula_one']=formula_split [:,0]
    df['formula_two']=formula_split [:,1]
    df = df.drop(['formula'],axis=1)
    zone= 'Asia/Shanghai'
    df['end_time']=pd.to_datetime(df['end_time'], utc=True).dt.tz_convert(zone)
    
    return df
    
def process_paper(df:pd.DataFrame)->pd.DataFrame:
    zone= 'Asia/Shanghai'
    df['time']=pd.to_datetime(df['time'], utc=True).dt.tz_convert(zone)
    
    
    class math_val():
        def __init__(self,args:list,target_args:list) -> None:
            self.args=args
            self.target_args=target_args
    
        def count_mean_var(self,row,targets):
            row=row[targets]
            mean=np.mean(row)
            var=np.var(row)
            return mean,var

        def process_df(self,df:pd.DataFrame)->pd.DataFrame:
            return_val=df.apply(self.count_mean_var,axis=1,args=(self.args,))
            return_val=np.array(return_val.to_list()).transpose(1,0)
            assert len(self.target_args)==return_val.shape[0],"target lenth error"
            for key,val in zip(self.target_args,return_val):
                df[key]=val
            return df
            
    # process KZ_PEOJPQD
    
    math_processor=math_val(['KZ_PEOJPQDL1','KZ_PEOWLBDL1','KZ_PEOWLBDL2'],['KZ_PEOJPQD_mean','KZ_PEOJPQD_var'])
    df=math_processor.process_df(df)
    
    
    return df



if __name__=="__main__":
    load_train_df()
    