import numpy as np
from tqdm import tqdm

class BlackJack:
    def __init__(self):
        # Actions
        self.action_hit = 0 
        self.action_strike = 1
        self.action = [self.action_hit, self.action_strike]
        
        # State
        self.state = []

        # player
        self.policy_player = np.zeros(22, dtype=np.int8)
        self.init_policy_player()
        self.player_sum = 0 # player 手中牌的和
        self.player_useable_ace = False # player 是否使用手中的 ace
        self.player_ace_count = 0 # player 手中 ace 数量
        self.player_trajectory = [] # 记录 player 当局的轨迹
        


        # dealer
        self.policy_dealer = np.zeros(22, dtype=np.int8)
        self.init_policy_dealer()
        self.dealer_card1 = 0 # dealer 初始手中牌一的值
        self.dealer_card2 = 0 # dealer 初始手中牌二的值
        self.dealer_sum = 0 # dealer 手中牌和
        self.dealer_ace_count = 0 # dealer 手中 ace 数量
        self.dealer_ace_useable = False # dealer 是否使用 ACE

        self.init_play()
    
    def init_policy_player(self):
        """
        初始化 player 的策略
        """
        for i in range(12, 20):
            self.policy_player[i] = self.action_hit
        self.policy_player[20] = self.action_strike
        self.policy_player[21] = self.action_strike
    
    def init_policy_dealer(self):
        """
        初始化 dealer 的策略
        """
        # 和小于16时，继续拿牌
        for i in range(12, 17):
            self.policy_dealer[i] = self.action_hit
        # 和大于16时，停止拿牌
        for i in range(17,22):
            self.policy_dealer[i] = self.action_strike

    def behavior_policy_player(self):
        """
        行动策略：等概率选择两种动作
        """
        if np.random.randint(1, 0.5) == 1:
            return self.action_strike
        return self.action_hit
    
    @classmethod
    def get_card():
        """
        随机发一张牌，ACE 记为 1
        """
        card_id = np.random.randint(1, 14)
        return card_id
    
    def card_value(card_id):
        """
        根据拿到的牌获得对应的价值，将 ACE 默认取值为 11
        """
        if card_id == 1:
            return 11
        elif card_id > 10:
            return 10
        else:
            return card_id
        
    def init_play(self,initial_state = None):
        if initial_state == None:
            # 初始化 player 的状态
            while self.player_sum < 12:
                card = self.get_card()
                self.player_sum  += self.card_value(card)

                # 判断是否值超过 21, 若超过必定是 11 + ACE
                # 两种可能，两张 ACE 或 一张 ACE
                if self.player_sum > 21:
                    self.player_sum -= 10

                # 判断拿到的牌是否为 ACE
                elif card == 1:
                    self.player_useable_ace = True

                # 初始化 dealer 的状态
                self.dealer_card1 = self.get_card()
                self.dealer_card2 = self.get_card()
                self.dealer_sum = self.card_value(self.dealer_card1) + self.card_value(self.dealer_card2)

                # 判断是否存在 ACE（至多一张 ACE 可用）
                if 1 in (self.dealer_card1, self.dealer_card2):
                    self.dealer_ace_useable = True
                
                if self.dealer_sum > 21:
                    self.dealer_sum -= 10

            assert self.dealer_sum <=21
            assert self.dealer_sum <=21
        else:
            self.player_useable_ace, self.player_sum, self.dealer_card1 = initial_state

        if self.player_useable_ace == True:
            self.player_ace_count =1
        
        if self.dealer_ace_useable == True:
            self.dealer_ace_count =1

        self.state = [self.player_useable_ace, self.player_sum, self.dealer_card1]

    def play(self, init_action=None):
        # player's turn
        while True:
            # 判断是否有指定初始动作
            if init_action is not None:
                action = init_action
                init_action = None
            else:
                action = self.policy_player[self.player_sum]

            self.player_trajectory.append([(self.player_useable_ace, self.player_sum, self.dealer_card1), action])

            # 停止抽牌，回合结束
            if action == self.action_strike:
                break

            # 否则重新抽一张牌
            card = self.get_card()
            self.player_sum += self.card_value(card)
            
            if card == 1:
                self.dealer_ace_count +=1

            while self.player_sum > 21 and self.player_ace_count:
                self.player_sum -= 10
                self.player_ace_count -= 1

            if self.player_sum > 21:
                return  self.state, -1, self.player_trajectory

            self.player_useable_ace = (self.player_ace_count == 1)

        # dealer's turn
        while True:
            action = self.policy_dealer[self.dealer_sum]

            if action == self.action_strike:
                break

            new_card = self.get_card()
            if new_card == 1:
                self.dealer_ace_count +=1

            self.dealer_sum += self.card_value(new_card)    

            while self.dealer_sum>21 and self.dealer_ace_count:
                self.dealer_sum -= 10
                self.dealer_ace_count -=1
            
            if self.dealer_sum > 21:
                return self.state, 1, self.player_trajectory
            
            self.dealer_ace_useable = (1 == self.dealer_ace_count)

            if self.player_sum > self.dealer_sum:
                return self.state, 1 ,self.player_trajectory
            elif self.player_sum == self.dealer_sum:
                return self.state, 0, self.player_trajectory
            else:
                return self.state, -1, self.player_trajectory

        

        


                
        