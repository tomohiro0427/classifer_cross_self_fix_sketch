import numpy as np

# バッチサイズとデータの初期化
b = 10  # バッチサイズ
y = np.random.randint(1, 5, size=(b,2,3))  # ランダムに値を生成 (1から4の範囲)
f = np.random.randint(1, 5, size=(b, 3, 5))  # [B, 246, 3]でランダムに値を生成

print("初期のy:", y)
print("初期のf[0]:", f[0])  # fの最初の要素のみ表示（全体は大きすぎるため）

# yの20%のバッチを0にする
threshold = 0.1
mask_20 = np.random.rand(b) < threshold
y[mask_20] = 0
# 同じ部分でfも0にする
f[mask_20] = 0

# 0でない部分のfに対して50%を0にする
non_zero_mask = ~mask_20  # yが0でない部分のマスク
f_non_zero = f[non_zero_mask,:,:]
print("yが0でない部分のf:", f_non_zero.shape)

# その部分の50%を0にする
threshold_f_50 = 0.4
mask_f_50 = np.random.rand(f_non_zero.shape[0]) < threshold_f_50
f_non_zero[mask_f_50] = 0
print("50%を0にした後のf:", f_non_zero.shape)

# 反対に、その対応するyの部分を50%を0にする
y_non_zero = y[non_zero_mask,:,:]
y_non_zero[~mask_f_50] = 0
print("50%を0にした後のy:", y_non_zero.shape)

# 更新したfとyを元に戻す
f[non_zero_mask,:,:] = f_non_zero
y[non_zero_mask,:,:] = y_non_zero


# 結果を表示
print("\n変更後のy:", y)
print("変更後のf[0]:", f)  # fの最初の要素のみ表示
