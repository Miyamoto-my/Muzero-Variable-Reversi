import datetime
import pathlib

import math
import numpy
import torch
import copy
import random

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt:オフ
        # More information is available here（詳細はこちら）: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # numpy、torch、およびゲームのシード
        self.max_num_gpus = None    # 使用する GPU の最大数を修正します。通常、十分なメモリがある場合は、単一の GPU を使用する (1 に設定する) 方が高速です。利用可能なすべての GPU を使用するものはありません
                                    
        ###  ゲーム
        self.observation_shape = (3, 10, 10)  # ゲーム観察の寸法は、3D (チャンネル、高さ、幅) でなければなりません。 1D 配列の場合は、(1, 1, 配列の長さ) に変形してください
        self.action_space = list(range((10 * 10) + 1))  # すべての可能なアクションの固定リスト。長さだけを編集する必要があります
        self.players = list(range(2))   # プレイヤーのリスト。長さだけを編集する必要があります
        self.stacked_observations = 0   # 現在の観測に追加する以前の観測と以前のアクションの数

        #  評価
        self.muzero_player = 0      # Muzero がプレイを開始するターン (0: MuZero が最初にプレイ、1: MuZero が 2 番目にプレイ)
        self.opponent = "expert"    # MuZeroがマルチプレイヤーゲームでの進捗状況を評価するために直面​​するハードコーディングされたエージェント。トレーニングには影響しません。 Game クラスで実装されている場合は、なし、「ランダム」または「エキスパート」
                                    

        ### セルフプレイ
        self.num_workers = 1                # リプレイ バッファを供給するために自己再生する同時スレッド/ワーカーの数
        self.selfplay_on_gpu = False
        self.max_moves = 101                # ゲームが終了していない場合の移動の最大数
        self.num_simulations = 400          # 自己シミュレートされた将来の動きの数
        self.discount = 1                   # 報酬の時系列割引
        self.temperature_threshold = None   # visit_softmax_temperature_fn で指定された温度を 0 に下げる (つまり、最適なアクションを選択する) までの移動回数。 None の場合、毎回 visit_softmax_temperature_fn が使用されます

        # ルート先行探索ノイズ
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCBスタイル
        self.pb_c_base = 19652
        self.pb_c_init = 1.25


        ### 通信網
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # 値と報酬は (ほぼ sqrt で) スケーリングされ、-support_size から support_size の範囲でベクトルにエンコードされます。 support_size <= sqrt(max(abs(割引報酬))) となるように選択してください

        # 残差のネットワーク
        self.downsample = False     # 表現ネットワークの前に観察をダウンサンプリング、False / "CNN" (ライター) / "resnet" (論文の付録ネットワーク アーキテクチャを参照)
        self.blocks = 6             # ResNet 内のブロック数
        self.channels = 128         # ResNet のチャンネル数
        self.reduced_channels_reward = 2    # リワードヘッドのチャンネル数
        self.reduced_channels_value = 2     # バリューヘッドのチャンネル数
        self.reduced_channels_policy = 4    # ポリシー ヘッドのチャネル数
        self.resnet_fc_reward_layers = [64] # 動的ネットワークの報酬ヘッドに隠れ層を定義する
        self.resnet_fc_value_layers = [64]  # 予測ネットワークの値の頭に隠れ層を定義します
        self.resnet_fc_policy_layers = [64] # 予測ネットワークのポリシー ヘッドで隠れ層を定義する

        # 完全に接続されたbulbネットワーク
        self.encoding_size = 32
        self.fc_representation_layers = [16]    # 表現ネットワークで隠れ層を定義する
        self.fc_dynamics_layers = [64]          # ダイナミクス ネットワークの隠れ層を定義する
        self.fc_reward_layers = [64]            # 報酬ネットワークの隠れ層を定義する
        self.fc_value_layers = [16]             # バリュー ネットワークの隠れ層を定義する
        self.fc_policy_layers = [16]            # ポリシー ネットワークの隠れ層を定義する


        ### トレーニング
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # モデルの重みと TensorBoard ログを保存するパス
        self.save_model = True          # チェックポイントを results_path に model.checkpoint として保存します
        self.training_steps = 10000     # トレーニング ステップの総数 (つまり、バッチに応じた重みの更新)
        self.batch_size = 512           # 各トレーニング ステップでトレーニングするゲームのパーツ数
        self.checkpoint_interval = 50   # 自動再生用モデルを使用する前のトレーニング ステップ数
        self.value_loss_weight = 1      # 価値関数のオーバーフィッティングを避けるために値の損失をスケーリングします。論文では 0.25 を推奨しています (論文の付録 Reanalyze を参照してください)。
        self.train_on_gpu = torch.cuda.is_available()   # 可能な場合は GPU でトレーニングする

        self.optimizer = "Adam"     #「アダム」または「SGD」。紙はSGDを使用
        self.weight_decay = 1e-4    # L2 重みの正則化
        self.momentum = 0.9         # オプティマイザが SGD の場合にのみ使用

        # Exponential learning rate schedule (指数関数的学習率スケジュール)
        self.lr_init = 0.003    # 初期学習率
        self.lr_decay_rate = 1  # 一定の学習率を使用するには、1 に設定します
        self.lr_decay_steps = 10000


        ### Replay Buffer リプレイバッファ
        self.replay_buffer_size = 10000  # リプレイ バッファに保持するセルフプレイ ゲームの数
        self.num_unroll_steps = 101     # バッチ要素ごとに保持するゲームの動きの数
        self.td_steps = 101             # 目標値を計算するために考慮する将来のステップ数
        self.PER = True                 # 優先リプレイ (紙の付録のトレーニングを参照)、ネットワークにとって予期しないリプレイ バッファ内の要素を優先的に選択します。
        self.PER_alpha = 0.5            # どのくらいの優先順位付けが使用されているか、0 は一様なケースに対応し、論文は 1 を示唆しています

        # Reanalyze (紙の付録 Reanalyse を参照)
        self.use_last_model_value = True    # 最後のモデルを使用して、より新鮮で安定した n ステップ値を提供します (紙の付録の再分析を参照してください)。
        self.reanalyse_on_gpu = False


        ### セルフプレイとトレーニングの比率を調整して、オーバーフィット/アンダーフィットを回避します
        self.self_play_delay = 0    # ゲームをプレイするたびに待機する秒数
        self.training_delay = 0     # 各トレーニング ステップの後に待機する秒数
        self.ratio = None           # 自己再生ステップ比ごとの望ましいトレーニングステップ。同期バージョンと同等で、トレーニングにはさらに長い時間がかかる場合があります。無効にするには、なしに設定します
        # fmt: オン

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        訪問回数の分布を変更して、トレーニングが進むにつれてアクションの選択がより貪欲になるようにするためのパラメーター。
        値が小さいほど、最適なアクション (つまり、訪問回数が最も多いアクション) が選択される可能性が高くなります。

        戻り値:
            正のフロート。
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

class Game(AbstractGame):
    """
    ゲーム ラッパー。
    """
    def __init__(self, seed=None):
        self.env = VariableReversi()

    def step(self, action):
        """
        ゲームにアクションを適用します。

        引数:
            action : 実行する action_space のアクション。

        戻り値:
            新しい観測、報酬、およびゲームが終了した場合のブール値。
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        現在のプレイヤーを返します。

        戻り値:
            現在のプレーヤーです。これは構成内のプレーヤー リストの要素である必要があります。
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        各ターンで法的措置を返す必要があります。利用できない場合は返すことができます
        アクション スペース全体。各ターンで、ゲームは返されたアクションの 1 つを処理できなければなりません。

        リーガル ムーブの計算が長すぎる複雑なゲームの場合、リーガル アクションを定義することが考えられます。
        アクション スペースと同じですが、アクションが違法な場合は負の報酬を返します。

        戻り値:
            整数の配列、アクション スペースのサブセット。
        """
        return self.env.legal_actions()

    def reset(self):
        """
        新しいゲームのためにゲームをリセットします。

        戻り値:
            ゲームの最初の観察。
        """
        return self.env.reset()

    def close(self):
        """
        ゲームを正しく終了します。
        """
        pass

    def render(self):
        """
        ゲームの観察結果を表示します。
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        マルチプレイヤー ゲームの場合は、ユーザーに法的措置を求める
        対応するアクション番号を返します。

        戻り値:
            アクション スペースからの整数。
        """
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action
    
    def expert_agent(self):
        """
        MuZero がマルチプレイヤー ゲームでの進捗状況を評価するために直面​​する、ハードコーディングされたエージェント。
        トレーニングには影響しません

        戻り値:
            現在のゲーム状態で実行する整数としてのアクション
        """
        return self.env.expert_action()
    
    def action_to_string(self, action):
        """
        アクション番号をアクションを表す文字列に変換します。
        引数:
            action_number: アクション スペースからの整数。
        戻り値:
            アクションを表す文字列。
        """
        return self.env.action_to_human_input(action)

class VariableReversi:
    def __init__(self):
        self.board_size = 10
        self.player = 1
        self.count = [2, 2]
        self.board_size_range = [4, 6, 8, 10]
        self.board = self.set_board()
        self._pass = self.board_size * self.board_size
    
    def set_board(self):
        """
        盤面を生成する
        -----
        戻り値
            new_board (int array): 盤面
        """
        def get_board_size():
            """
            可変式盤面の幅を生成する
            -----
            戻り値
                (int): 4、6、8、10、の何れかをランダムで返す
            """
            return random.choice(self.board_size_range)
        
        def set_center_koma(active_board):
            """ 盤面の中央4マスに白及び黒を配置する """
            base_y = math.floor(self.height / 2)
            base_x = math.floor(self.width / 2)

            active_board[base_y-1][base_x-1] = 1
            active_board[base_y][base_x] = 1
            active_board[base_y-1][base_x] = -1
            active_board[base_y][base_x-1] = -1        

        self.height = get_board_size()
        self.width = get_board_size()

        all_board = numpy.full((self.board_size, self.board_size), 999, dtype="int32")
        active_board = numpy.zeros((self.height, self.width), dtype="int32")
        set_center_koma(active_board)
        
        w1, h1 = all_board.shape
        w2, h2 = active_board.shape
        w, h = (w1-w2)//2, (h1-h2)//2
        new_board = copy.deepcopy(all_board)
        new_board[w:(-w if w else None), h:(-h if h else None)] = active_board
        return new_board

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = self.set_board()
        self.player = 1
        return self.get_observation()
    
    def step(self, action):
        """
        アクションをゲームに適用する
        ------
        引数
            action (int):プレイヤーの行動
        ------
        戻り値
            observation (int array), reward (int), done (True / False)
        """
        if action != self._pass:
            y = math.floor(action / self.board_size)
            x = action % self.board_size
            reverse = self.is_legal_action_yx(y, x)[1]
            self.board[y][x] = self.player
            for r in reverse:
                ry = math.floor(r / self.board_size)
                rx = r % self.board_size
                self.board[ry][rx] = self.player
                
            self.count[0] = numpy.count_nonzero(self.board == 1)
            self.count[1] = numpy.count_nonzero(self.board == -1)
        
        done = self.is_finished()
        reward = 1 if done else 0
        self.player *= -1
        return self.get_observation(), reward, done

    def get_observation(self):
        """
        observation観測データを返す
        ------
        戻り値
            board_player1 (float array): プレイヤー1視点の盤面, 
            board_player2 (float array): プレイヤー2視点の盤面, 
            board_to_play (int array): 現在のプレイヤー番号, 
        """
        board_player1 = numpy.where(self.board == -1, 0.0, self.board)
        board_cp = numpy.where(self.board == 1, 0.0, self.board)
        board_player2 = numpy.where(board_cp == -1, 1.0, board_cp)
        board_to_play = numpy.full((self.board_size, self.board_size), self.player, dtype="int32")
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        """
        合法手リストを返す
        -----
        戻り値
            legal (int array): 合法手リスト、合法手がない場合（パスをする場合）は_passを返す
        """
        legal = []
        for y in range(0, self.board_size):
            for x in range(0, self.board_size):
                if self.is_legal_action_yx(y, x)[0]:
                    legal.append(y * self.board_size + x)
        
        if legal == []:
            return [self._pass]
        return legal

    def is_legal_action_yx(self, y, x):
        """
        入力値の合法手の有無と裏返す配列を返す
        -----
        引数
            y (int): boardの一次元の要素番号
            x (int): boardの二次元の要素番号
        戻り値
            True or False: 合法手の有無
            flip (int array): 裏返すリスト
        """
        search_args_y = [0, 0, -1, 1, -1, 1, -1, 1]
        search_args_x = [-1, 1, 0, 0, -1, -1, 1, 1]
        flip = []

        if self.board[y][x] == 0:   # 入力値が盤面上置くことができるか
            for dy, dx in zip(search_args_y, search_args_x):
                py, px = y, x       # 現在の探索位置
                reverse_list = []   # 裏返す
                while True:
                    py, px = py+dy, px+dx
                    if (py < 0) or (self.board_size-1 < py) or (px < 0) or (self.board_size-1 < px):
                        break
                    p_state = self.board[py][px]
                    if (p_state == 0) or (p_state == 999):
                        break
                    elif (p_state == self.player):
                        if reverse_list == []:
                            break
                        else:
                            flip.extend(reverse_list)
                            break
                    else:
                        reverse_list.append(py * self.board_size + px)

        if flip != []:
            return True, flip
        return False, []

    def is_finished(self):
        """
        終了条件
        二回連続合法手がない且つ勝者の番で終了する
        -----
        戻り値
            True or False: 終了/続く
        """
        legal_action = []
        for _ in range(2):
            legal_action.extend(self.legal_actions())
            self.player *= -1

        if (
            legal_action[0] == self._pass 
            and legal_action[1] == self._pass 
            and self.count[self.to_play()] == max(self.count)
        ):
            return True
        return False

    def render(self):
        """ 表示 """
        print(f"○: {self.count[0]}, ●: {self.count[1]}")
        print("○" if self.player == 1 else "●", "'s Turn")
        marker = "  "
        rs, cs = math.floor((self.board_size-self.height)/2), math.floor((self.board_size-self.width)/2)
        re, ce = rs + self.height, cs + self.width
        for i in range(self.width):
            marker = marker + str(i) + " "
        print(marker)
        for row in range(rs, re):
            print(str(row-rs), end=" ")
            for col in range(cs, ce):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end=" ")
                elif ch == 1:
                    print("○", end=" ")
                elif ch == -1:
                    print("●", end=" ")
            print()

    def human_input_to_action(self):
        """
        人が行動選択を行う
        -----
        戻り値
            True or False: 入力値の正否
            action (int): 行動値、パスの場合は_passを返す 
        """
        human_input = input("Enter an action. [yx] :")
        if (
            len(human_input) == 2
            and int(human_input[0]) in range(self.height)
            and int(human_input[1]) in range(self.width)
        ):
            y = int(human_input[0]) + math.floor((self.board_size - self.height) / 2)
            x = int(human_input[1]) + math.floor((self.board_size - self.width) / 2)
            if self.is_legal_action_yx(y, x)[0]:
                return True, y*self.board_size+x

        if sum(self.legal_actions()) == self._pass: # 入力値が不適切だった場合は、合法手の探索を行いパスの判定を行う
            return True, self._pass
        return False, -1

    def expert_action(self):
        return numpy.random.choice(self.legal_actions())

    def action_to_human_input(self, action):
        """
        アクションをユーザー表示文字列に変換する
        ------
        引数
            action (int): プレイヤーの行動
        ------
        戻り値
            (str): 表示文字列
        """
        if action == self._pass:    # アクションがパスの場合，文字列"pass"を返す
            return "pass"
            
        y = math.floor(action / self.board_size)
        x = action % self.board_size
        y -= math.floor((self.board_size-self.height)/2)
        x -= math.floor((self.board_size-self.width)/2)
        return str(y)+str(x)
