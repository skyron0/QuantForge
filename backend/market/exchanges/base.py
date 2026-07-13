from abc import ABC, abstractmethod


class BaseExchange(ABC):

    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def subscribe(self, symbol: str):
        pass

    @abstractmethod
    async def get_tick(self):
        pass