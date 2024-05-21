import logging
from decimal import Decimal
from typing import Dict, List

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class PMMAvallanedaStoikov(ScriptStrategyBase):
    """
    BotCamp - Market Making Strategies
    Description:
    The bot extends the Simple PMM example script by incorporating the Candles Feed and
    creating a custom status function that displays it.
    """
    bid_spread = 0.0001
    ask_spread = 0.0001
    order_refresh_time = 15
    order_amount = 0.01
    create_timestamp = 0
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    # Here you can use for example the LastTrade price to use in your strategy
    price_source = PriceType.MidPrice

    # Candles params
    candle_exchange = "binance"
    candles_interval = "1m"
    candles_length = 30
    max_records = 1000

    # added by dingsen
    # SET BY USERS
    target_base_quote_ratio = Decimal(0.5)  # user's target inventory percent, set by users
    # gamma is the inventory rist parameter, set by users;
    # gamma=1 -> very risk-averse -> want to go back to target as quick as possible
    # gamma=0 -> vert risky -> I don't care about inventory risk.(Note we should not set gamma=0 in this simple
    # a&s script as we need to divide gamma).
    gamma = Decimal(0.5)

    # CALCULATE IN SCRIPT OR FETCH FROM SOURCE
    time_left = Decimal(1)

    # overwrite by script
    current_base_quote_ration = 0.5

    # q is the inventory position deviation,
    # calculated based on user's choice of target_inventory_percent and users' current inventory
    q = 0

    # sigma is the market volatility, here we're using NATR as the volatility indicator, fetch from source
    sigma = 0

    # kappa is the order book liquidity, here we're using 1 as the order book liquidity indicator, calculated by script
    kappa = 1

    # reference price, calculated by script
    reference_price = 0

    # optimal spread, calculated by script
    optimal_spread = 0.0002

    # original price, fetch from source
    orig_price = 0

    # here we recognize base & quote. ETH is base and USDT is quote.
    base, quote = trading_pair.split('-')

    # Initializes candles
    candles = CandlesFactory.get_candle(CandlesConfig(connector=candle_exchange,
                                                      trading_pair=trading_pair,
                                                      interval=candles_interval,
                                                      max_records=max_records))

    # markets defines which order books (exchange / pair) to connect to. At least one exchange/pair needs to be
    # instantiated
    markets = {exchange: {trading_pair}}

    # start the candles when the script starts
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.candles.start()

    # stop the candles when the script stops
    def on_stop(self):
        self.candles.stop()

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            self.update_multipliers()
            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.proposal_checker(proposal)
            # proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp

    def get_time_left(self):
        """
        here we compute T-t, which is the time left. However, for cryptocurrencies the market is always on so
        it makes sense to set the time_left to be a constant but I do believe there should be some cyclic manner
        in crypto market so that this time_left variable can be used to tune the equation.

        TODO: find some way to consider cyclic behavior here.
        """
        self.time_left = 1
        return self.time_left

    def calculate_inventory_deviation(self):
        """
        author: dingsen
        This function calculates the inventory deviation parameter $q$.
        trader is short of base -> should buy more -> raising the reserved price -> q < 0
        trader is long of base -> should sell more -> decreasing the reserved price -> q > 0
        The paper suggests a complex way to deduce q that I don't understand lol. I'll use what Michael
        used in inventory shift video as q here.

        TODO: Understand what's going on in the paper and calculated q based on that.
        """
        base_balance = self.connectors[self.exchange].get_balance(self.base)
        base_price_in_quote = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        base_balance_in_quote = base_balance * base_price_in_quote
        quote_balance = self.connectors[self.exchange].get_balance(self.quote)
        self.current_base_quote_ration = Decimal(base_balance_in_quote / quote_balance)

        # msg = (f"base_balance: {base_balance}, base_price_in_quote:{base_price_in_quote}, "
        #        f"base_balance_in_quote:{base_balance_in_quote} quote_balance: {quote_balance}")
        # Note that this is different from Michael's video in that when we haven't reached the target ration,
        # we want to buy more and q should be < 0, which raising the buy price. That's why there is a negative at the
        # front.

        # divide by 100 to normalize
        self.q = Decimal(-1 * (self.target_base_quote_ratio -
                               self.current_base_quote_ration) / self.target_base_quote_ratio / 100)
        return self.q

    def calculate_order_book_liquidity(self):
        """
        author: dingsen
        This function calculate the order book liquidity parameter $gamma$.
        higher kappa -> more liquid -> smaller spread to be competitive to hit orders.
        lower kappa -> less liquid -> larger spread to make more profits.
        In the original paper, it chose 3 different liquidity parameters: 0.01, 0.1 and 1. Since in this example,
        We are using ETH-USDT as our asset pair, which is very liquid, I am going to use 1 as $gamma$ for now.

        TODO: The choice of $kappa$ can be chosen in better way?
        """
        self.kappa = 1
        return self.kappa

    def calculate_market_volatility(self):
        """
        author: dingsen
        This function calculate the market volatility parameter $sigma$.
        In the simple version of A&S impl, this function just fetches the NATR value of last order
        Not used in this script
        """
        return self.sigma

    def calculate_reservation_price(self):
        """
        author: dingsen
        This function calculate the reservation price.
        return: the reservation price
        formula: r=s-q*gamma*sigma^2*(T-t)
        """
        res = self.orig_price - self.q * self.gamma * self.sigma ** 2 * self.time_left * 100
        return res

    def calculate_optimal_spread(self):
        """
        author: dingsen
        This function calculate the optimal spread.
        formula: gamma * sigma^2 * time_left + 2/gamma * ln(1 + self.gamma / self.kappa)
        return: the optimal spread
        """
        res = self.gamma * self.sigma ** 2 * self.time_left
        res += 2 * Decimal(1 + self.gamma / self.kappa).ln() / self.gamma
        return res

    def get_candles_with_features(self):
        candles_df = self.candles.candles_df
        # dingsen: here we are fetching the NATR as a signal of volatility
        candles_df.ta.natr(length=self.candles_length, scalar=1, append=True)
        self.sigma = Decimal(candles_df[f"NATR_{self.candles_length}"].iloc[-1])  # set gamma to be NATR
        return candles_df

    def update_multipliers(self):
        """
        This function update the spread & reference price
        """

        self.orig_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        # FIXME: Here we are using mid-price as the price source, which should be adjusted
        self.calculate_inventory_deviation()
        self.reference_price = self.calculate_reservation_price()
        self.optimal_spread = self.calculate_optimal_spread()
        self.bid_spread = self.optimal_spread / 2 / self.reference_price
        self.ask_spread = self.optimal_spread / 2 / self.reference_price

    def create_proposal(self) -> List[OrderCandidate]:
        """
        This function creates potential orders based on the parameters we updated in update_multipliers
        """
        ref_price = self.reference_price
        buy_price = ref_price * Decimal(1 - self.bid_spread)
        sell_price = ref_price * Decimal(1 + self.ask_spread)

        buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                   order_side=TradeType.BUY, amount=Decimal(self.order_amount), price=buy_price)

        sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.SELL, amount=Decimal(self.order_amount), price=sell_price)

        return [buy_order, sell_order]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)
        return proposal_adjusted

    def proposal_checker(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        return self.adjust_proposal_to_budget(proposal)

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                      order_type=order.order_type, price=order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                     order_type=order.order_type, price=order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = (
            f"{event.trade_type.name} {round(event.amount, 2)}"
            f" {event.trading_pair} {self.exchange} at {round(event.price, 2)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def format_status(self) -> str:
        """
        Returns status of the current strategy and displays candles feed info
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])

        ref_price = self.reference_price
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
        best_bid_spread = (ref_price - best_bid) / ref_price
        best_ask_spread = (best_ask - ref_price) / ref_price

        lines.extend(["\n----------------------------------------------------------------------\n"])

        lines.extend(["  Spreads:"])
        lines.extend([f"  Bid Spread (bps): {self.bid_spread * 10000:.4f} |"
                      f" Best Bid Spread (bps): {best_bid_spread * 10000:.4f}"])
        lines.extend([f"  Ask Spread (bps): {self.ask_spread * 10000:.4f} |"
                      f" Best Ask Spread (bps): {best_ask_spread * 10000:.4f}"])
        lines.extend([f"  optimal spread: {self.optimal_spread}"])
        lines.extend([f"  inventory deviation = q: {self.q}"])
        lines.extend([f"  inventory risk = gamma:{self.gamma}"])
        lines.extend([f"  market volatility = sigma:{self.sigma}"])
        lines.extend([f"  T-t=time_left:{self.time_left}"])
        lines.extend([f"  orderbook liquidity = kappa:{self.kappa}"])
        lines.extend([f"  Orig Price: {self.orig_price:.4f} | Reference Price: {self.reference_price:.4f}"])
        lines.extend(["\n----------------------------------------------------------------------\n"])

        candles_df = self.get_candles_with_features()
        lines.extend([f"  Candles: {self.candles.name} | Interval: {self.candles.interval}", ""])
        lines.extend(["    " + line for line in
                      candles_df.tail(self.candles_length).iloc[::-1].to_string(index=False).split("\n")])

        return "\n".join(lines)
