# Forex-TrendCatcher
Målet är att bygga en strategi som utnyttjar trend expansioner i mer volatila valutapar som tenderar att trenda mer än andra valutapar. Vi kommer testa olika strategier i början som följer detta spåret till vi har något med edge. Vi kommer börja testa på 1 timmes tids-
intervall på USDJPY på in sample data 2012 - 2020. Första idén är att köpa när pris bryter ovanför asia range och säljan när det bryter under. Vi testar detta med exits baserat på momentumförlust, t.ex. ATR, range eller trendindikator som ADX. Kanske testar ATR entry 
istället för asia range också. Eftersom ATR kommer skilja mellan olika valutapar och vara relativt bör vi inte använda ett absolut värde i logiken, utan istället använda över/under medelvärdet av ATR som krav på exit eller entry. Nu kör vi första testet, buy stop vid asia high och sell stop vid asia low. Long exits när atr går under medelvärdet av de senaste 50 atr värdena, så en 50 moving average på atr. Vi sätter också då att atr måste vara över atr 
moving average vid entry för att logiken ska hänga ihop, annars kan det bli att många entries sker men sedan stängs direkt. Vi definierar asia session
som 00:00-07:59. Vi sätter vårat entry handelsfönster till London session 08:00-11:59. Exits ska kunna ske utanför detta fönster. Vi låser även entries
till row["close"] > row["asia_high"] and prev_row["close"] <= prev_row["asia_high"], annars kan vi få en exit och sedan ny entry när pris redan expanderat.
Vi testar detta på USDJPY in sample 2012-2020. vi testade att köra entrylogiken bullish_breakout = close_price > asia_high och fick 
Market: USDJPY 
Trades: 401 
Total PnL (points): 14.8969 
Gross Profit: 83.9098 
Gross Loss: -69.0129 
Profit Factor: 1.2159 
Winrate: 0.4788 
Avg Win: 0.4370 
Avg Loss: -0.3302 
Expectancy (avg/trade): 0.0371 
Max Drawdown (points): 9.4751 
Max Losing Streak (trades): 12 
Sharpe (trade-level): 1.2021 
När vi ändrade till bullish_breakout = close_price > asia_high and prev_close <= prev_asia_high fick vi: 
Market: USDJPY 
Trades: 253 
Total PnL (points): 0.2887 
Gross Profit: 47.3219 
Gross Loss: -47.0332 
Profit Factor: 1.0061 
Winrate: 0.4783 
Avg Win: 0.3911 
Avg Loss: -0.3563 
Expectancy (avg/trade): 0.0011 
Max Drawdown (points): 9.2755 
Max Losing Streak (trades): 9 
Sharpe (trade-level): 0.0299
Detta säger att edgen inte finns i själv crossovern över/under asia rangen, utan att om pris redan är etablerat ovanför/under rangen och volatiliteten(ATR14) fortfarande stöder rörelsen så finns edge. Edgen är alltså mer regime continuation än breakout. Vi har alltså hittat en strategi
som utnyttjar köp styrka ovanför rangen. Vi har nu en strategi vi kan börja jobba med. Vi fortsätter med optimering, stress tester och out of sample
i projektloggen för Forex-TrendCatcher.
