import pandas as pd   # 0.19.2
import numpy as np    # 1.11.3
import matplotlib.pyplot as plt  # 2.0.0
import seaborn as sns # 0.7.1

# Global constants and dataframes 
DAYS_IN_TRADING_YEAR = 252
start_date = '2010-01-01'
end_date = '2015-12-31'
data = pd.read_csv('sp100.csv',index_col=0,parse_dates=True)
data = data[start_date:end_date].dropna(axis=1) # note this introduces survivorship bias
data.columns = [val.replace(' US Equity','') for val in data.columns]
rtndf = data.pct_change().dropna()

def mc_sim(stat_type,mcruns=1000,tms_port_size=10):
    '''
    stat_type: rtn    --  histograms of expected returns
               vol    --  histograms of historical vol
               maxdd  --  histograms of max drawdown
               sr     --  histograms of Sharpe Ratios
               sr_tms --  time series of rolling Sharpe Ratios

    mc_runs -- Number of runs in Monte Carlo Simulation
    tms_port_size -- Number of securities to select for portfolios in the 
                     rolling Sharpe Ratio example.
    '''

    ann_fact = float(len(data))/DAYS_IN_TRADING_YEAR
    num_stocks = len(rtndf.columns)
    stat = []

    port_size = [95,50,20,10,5,2] # Size of portfolios in simulation 

    plt.figure(figsize=(11,10)) 

    # Main Monte Carlo simulation
    for rand_port_size in port_size:
        if stat_type == "sr_tms":
            port_size = tms_port_size
        for _ in xrange(mcruns):
            stock_inds = np.random.choice(xrange(num_stocks),size=rand_port_size,replace=False)
            tmprtndf = rtndf.ix[:,stock_inds]
            wgts = np.random.rand(rand_port_size)
            wgts = wgts/wgts.sum()
            rand_port = tmprtndf.dot(wgts)
            if stat_type == 'rtn':
                stat.append(rand_port.cumsum()[-1]/ann_fact) # annualized return
                title_str = 'Annualized Return'
                plt.xlim([0.05,0.25])
            elif stat_type == 'vol':
                stat.append(np.std(rand_port)*np.sqrt(DAYS_IN_TRADING_YEAR))
                title_str = 'Annualized Volatility'
                plt.xlim([0.1,0.25])
            elif stat_type == 'maxdd':
                cumrtn = rand_port.cumsum()
                i = np.argmax(np.maximum.accumulate(cumrtn) - cumrtn) # end of the period
                j = np.argmax(cumrtn[:i]) # start of period
                stat.append(cumrtn[j]-cumrtn[i])
                title_str = 'Maximum Drawdown'
                plt.xlim([0.05,0.4])
            elif stat_type == 'sr':
                ann_rtn = rand_port.cumsum()[-1]/ann_fact 
                vol = np.std(rand_port)*np.sqrt(DAYS_IN_TRADING_YEAR)
                stat.append(ann_rtn/vol)
                title_str = 'Annualized Sharpe Ratio'
                plt.xlim([0.15,1.5])
            elif stat_type == 'sr_tms':
                rtnsrs = rand_port.rolling(DAYS_IN_TRADING_YEAR).sum()            
                volsrs = rand_port.rolling(DAYS_IN_TRADING_YEAR).std()*np.sqrt(DAYS_IN_TRADING_YEAR)
                srsrs = rtnsrs/volsrs
                stat.append(srsrs.dropna())
                title_str = 'Sharpe Ratio Series'
            else:
                raise ValueError('Unrecognized stat_type ' + stat_type)
    
        statarr = np.array(stat)

        if stat_type == "sr_tms":
            break
        else:
            sns.distplot(statarr,hist=False,label=str(rand_port_size))

    if stat_type == 'sr_tms':
        mean_srs = np.mean(statarr,axis=0)
        std_srs = np.std(statarr,axis=0)
        plt.plot(stat[0].index,mean_srs)
        plt.plot(stat[0].index,mean_srs+2*std_srs,'k',linewidth=1.0,alpha=0.3)
        plt.plot(stat[0].index,mean_srs-2*std_srs,'k',linewidth=1.0,alpha=0.3)
        plt.ylabel('Sharpe Ratio')
        plt.title('Year over Year Annualized Sharpe Ratio for Portfolios of ' + str(tms_port_size) + ' Randomly Selected\n SP100 Stocks with 2-Sigma Error Curves (Grey)')
    else:
        plt.legend(title='Number of\nSecurities\nin Subset')
        plt.title(title_str + ' of Random Portfolios of Random Subsets of\n SP100 Stocks from ' + start_date + ' to ' + end_date)
        plt.xlabel(title_str)
        plt.ylabel('Normalized Histogram Value')

        if stat_type != 'sr':
            plt.gca().set_xticklabels(['{0:3.1f}%'.format(100*x) for x in plt.gca().get_xticks()])
        else:
            plt.gca().set_xticklabels(['{0:3.1f}'.format(x) for x in plt.gca().get_xticks()])

if __name__ == "__main__":
    stat_type_list = ['rtn','vol','maxdd','sr','sr_tms']
    for stat_type in stat_type_list:
        mc_sim(stat_type,mcruns=10000,tms_port_size=10)
    plt.show()
