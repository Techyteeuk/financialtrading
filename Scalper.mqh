// Scalper for a small account on a 5-minute chart

// Define variables
input int Period1 = 5; // Period for the first moving average
input int Period2 = 10; // Period for the second moving average
input int PivotRange = 3; // Range for the pivot point
double PivotPoint; // Pivot point
double MovingAvg1; // First moving average
double MovingAvg2; // Second moving average

// Calculate the pivot point
PivotPoint = (High[PivotRange] + Low[PivotRange] + Close[PivotRange]) / 3;

// Calculate the first moving average
MovingAvg1 = iMA(NULL, 0, Period1, 0, MODE_EMA, PRICE_CLOSE, 0);

// Calculate the second moving average
MovingAvg2 = iMA(NULL, 0, Period2, 0, MODE_EMA, PRICE_CLOSE, 0);

// Plot the pivot point
Plot(PivotPoint, "Pivot Point", White, STYLE_DOT);

// Plot the moving averages
Plot(MovingAvg1, "Moving Avg 1", Yellow, STYLE_SOLID);
Plot(MovingAvg2, "Moving Avg 2", Blue, STYLE_SOLID);

// Identify entry and exit points based on the pivot point and moving average cross
if (MovingAvg1 > MovingAvg2 && Close[0] > PivotPoint)
  return 1; // Long entry
else if (MovingAvg1 < MovingAvg2 && Close[0] < PivotPoint)
  return -1; // Short entry
else if (MovingAvg1 > MovingAvg2 && Close[0] < PivotPoint)
  return 0; // Long exit
else if (MovingAvg1 < MovingAvg2 && Close[0] > PivotPoint)
  return 0; // Short exit
else
  return 0; // Neutral
