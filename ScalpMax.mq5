// Scalp Max for a small account on a 5-minute chart

// Custom indicator for a scalping strategy on a 5-minute chart
#include <Scalper.mqh>

// Define variables
extern int StopLoss = 20; // Stop loss in pips
extern int TakeProfit = 10; // Take profit in pips
extern int TrailingStop = 10; // Trailing stop in pips
Scalper indicator;

// Initialize the custom indicator
int OnInit()
{
  indicator.SetPeriod1(5);
  indicator.SetPeriod2(10);
  indicator.SetPivotRange(3);
  return(INIT_SUCCEEDED);
}

// Place trades based on the custom indicator
void OnTick()
{
  // Identify entry and exit points based on the custom indicator
  int result = indicator.iCustom(NULL, 0, "Scalper", 0, 0);
  
  // Place a long (buy) or short (sell) order depending on the result
  if (result == 1)
  {
    // Place a long order with a stop loss and take profit
    OrderSend(Symbol(), OP_BUY, 0.1, Ask, 3, Bid - StopLoss * Point, Ask + TakeProfit * Point);
    
    // Set a trailing stop
    OrderModify(OrderTicket(), OrderOpenPrice(), Bid - TrailingStop * Point, OrderTakeProfit(), 0, Blue);
  }
  else if (result == -1)
  {
    // Place a short order with a stop loss and take profit
    OrderSend(Symbol(), OP_SELL, 0.1, Bid, 3, Ask + StopLoss * Point, Bid - TakeProfit * Point);
    
    // Set a trailing stop
    OrderModify(OrderTicket(), OrderOpenPrice(), Ask + TrailingStop * Point, OrderTakeProfit(), 0, Red);
  }
}
