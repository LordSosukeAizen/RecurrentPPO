import gymnasium as gym
import numpy as np
import yfinance as yf
import pandas as pd
import pygame
import time
from helpers import random_only_date



class StockEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        ticker: str,
        start_date,
        end_date,
        cash=10000.0,
        change_env=False,
        render_mode="human",
    ):
        super().__init__()

        self.ticker = ticker
        self.render_mode = render_mode
        self.init_start_date = start_date
        self.init_end_date = end_date
        self.init_cash = cash
        self.change_env = change_env

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = gym.spaces.Discrete(3)

        # Observation: [shares_held, current_price]
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(2,), dtype=np.float32
        )

        # Load data
        self._fetch_data(self.init_start_date, self.init_end_date)

        # --- Rendering setup ---
        if self.render_mode == "human":
            pygame.init()
            self.width, self.height = 800, 400
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(f"StockEnv: {self.ticker}")
            self.clock = pygame.time.Clock()

        self.reset()

    # ------------------------------------------------------------------

    def _fetch_data(self, start, end):
        df = yf.download(self.ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.swaplevel(axis=1)[self.ticker]

        self._close = df["Close"].dropna()
        self._len = len(self._close)

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cash = float(self.init_cash)
        self.stocks_hold = 0
        self.tick = 0

        # Rendering history
        self.prices = []
        self.buy_x, self.buy_y = [], []
        self.sell_x, self.sell_y = [], []

        if self.change_env:
            start = random_only_date(self.init_start_date, self.init_end_date)
            self._fetch_data(start, self.init_end_date)

        price = self._close.iloc[self.tick]
        state = np.array([self.stocks_hold, price], dtype=np.float32)

        return state, {}

    # ------------------------------------------------------------------

    def step(self, action: int):
        assert self.action_space.contains(action)

        price = self._close.iloc[self.tick]
        prev_value = self.cash + self.stocks_hold * price

        # Log price for rendering (time-aligned)
        if self.render_mode == "human":
            self.prices.append(price)

        # --- Execute action ---
        if action == 1:  # Buy
            if self.cash >= price:
                self.cash -= price
                self.stocks_hold += 1
                if self.render_mode == "human":
                    self.buy_x.append(self.tick)
                    self.buy_y.append(price)

        elif action == 2:  # Sell
            if self.stocks_hold > 0:
                self.cash += price
                self.stocks_hold -= 1
                if self.render_mode == "human":
                    self.sell_x.append(self.tick)
                    self.sell_y.append(price)

        # Advance time
        self.tick += 1

        terminated = self.tick >= self._len - 1
        truncated = False

        next_price = self._close.iloc[self.tick]
        curr_value = self.cash + self.stocks_hold * next_price

        # Log-return reward (stable & scale-invariant)
        reward = curr_value - prev_value

        state = np.array([self.stocks_hold, next_price], dtype=np.float32)

        return state, round(reward, 2), terminated, truncated, {}

        # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return

        self.clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((255, 255, 255))

        if len(self.prices) < 2:
            pygame.display.flip()
            return

        prices = np.array(self.prices)
        pmin, pmax = prices.min(), prices.max()

        def scale_y(p):
            return int(
                self.height - 50 - (p - pmin) / (pmax - pmin + 1e-6) * (self.height - 100)
            )

        # Draw price line
        for i in range(1, len(prices)):
            x1 = int((i - 1) / len(prices) * self.width)
            x2 = int(i / len(prices) * self.width)
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (x1, scale_y(prices[i - 1])),
                (x2, scale_y(prices[i])),
                2,
            )

        # Draw buy markers
        for x, y in zip(self.buy_x, self.buy_y):
            px = int(x / len(prices) * self.width)
            py = scale_y(y)
            pygame.draw.polygon(
                self.screen, (0, 200, 0),
                [(px, py - 6), (px - 5, py + 5), (px + 5, py + 5)]
            )

        # Draw sell markers
        for x, y in zip(self.sell_x, self.sell_y):
            px = int(x / len(prices) * self.width)
            py = scale_y(y)
            pygame.draw.polygon(
                self.screen, (200, 0, 0),
                [(px, py + 6), (px - 5, py - 5), (px + 5, py - 5)]
            )

        pygame.display.flip()


    # ------------------------------------------------------------------
    def close(self):
        if self.render_mode == "human":
            import pygame.surfarray
            from PIL import Image

            # Grab pixels from screen
            arr = pygame.surfarray.array3d(self.screen)

            # pygame uses (width, height, channels)
            # PIL expects (height, width, channels)
            arr = arr.swapaxes(0, 1)

            img = Image.fromarray(arr)
            img.save(f"{self.ticker}_final.png")

            pygame.quit()




