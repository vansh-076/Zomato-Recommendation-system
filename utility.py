import pandas as pd
import os 

def preprocess(path,country_code_path):
    df=pd.read_csv(path,encoding='latin-1')
    country_df=pd.read_excel(country_code_path)
    merge_df=df.merge(country_df,on='Country Code',how='left')
    merge_df.drop(columns=['Country Code','Is delivering now','Switch to order menu','Rating color','Locality Verbose','Currency'],inplace=True)
    merge_df.dropna(axis=0,inplace=True)
    merge_df.drop_duplicates(inplace=True)
    merge_df=merge_df.loc[merge_df['Country']=='India']
    merge_df.to_csv("final_dataset.csv")

if __name__=="__main__":
    path=os.path.join(os.getcwd(),"zomato.csv")
    country_code_path=os.path.join(os.getcwd(),"Country-Code.xlsx")
    preprocess(path,country_code_path)