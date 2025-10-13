#!/usr/bin/env python3
"""
API测试脚本
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

sp500_top100 = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'GOOG', 'AMZN', 'META', 'AVGO', 'TSLA', 'ORCL', 'JPM', 'WMT', 'LLY',
                'V', 'NFLX', 'MA', 'XOM', 'JNJ', 'PLTR', 'COST', 'ABBV', 'HD', 'BAC', 'PG', 'AMD', 'UNH', 'GE', 'CVX',
                'KO', 'CSCO', 'IBM', 'TMUS', 'PM', 'WFC', 'MS', 'GS', 'ABT', 'CAT', 'CRM', 'AXP', 'MRK', 'LIN', 'MCD',
                'RTX', 'PEP', 'MU', 'TMO', 'DIS', 'UBER', 'ANET', 'APP', 'BX', 'T', 'NOW', 'INTU', 'BLK', 'INTC', 'C',
                'NEE', 'VZ', 'BKNG', 'AMAT', 'SCHW', 'QCOM', 'LRCX', 'GEV', 'BA', 'TJX', 'AMGN', 'TXN', 'ISRG', 'ACN',
                'APH', 'SPGI', 'GILD', 'DHR', 'ETN', 'BSX', 'ADBE', 'PANW', 'PFE', 'PGR', 'SYK', 'UNP', 'LOW', 'COF',
                'KLAC', 'HON', 'CRWD', 'HOOD', 'MDT', 'DE', 'LMT', 'IBKR', 'ADP', 'CEG', 'DASH', 'CB', 'MO', 'WELL']


def test_health():
    """测试健康检查"""
    print("🔍 测试健康检查...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_user_profile():
    """测试用户画像"""
    print("👤 测试用户画像...")

    # 初始化用户画像
    response = requests.post(f"{BASE_URL}/api/users/profile/init?user_id=test_user")
    print(f"初始化用户画像: {response.status_code}")
    if response.status_code == 200:
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_stock_data():
    """测试股票数据"""
    print("📈 测试股票数据...")

    # 分别测试每个股票，而不是一次性传多个
    # symbols = ["AAPL", "MSFT", "GOOGL"]

    response = requests.post(f"{BASE_URL}/api/stocks/fetch-raw-data?symbols={','.join(sp500_top100)}")
    print(f"获取原始股票数据: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f" 相应: {result['message']}")

    print()

    # 更新股票向量
    response = requests.post(f"{BASE_URL}/api/stocks/update-vectors?symbols={','.join(sp500_top100)}")
    print(f"更新股票向量: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print()


def test_recommendation():
    """测试推荐功能"""
    print("🎯 测试推荐功能...")

    response = requests.get(f"{BASE_URL}/api/stocks/recommend?user_id=test_user&top_k=3")
    print(f"获取推荐: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"推荐数量: {result['count']}")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec['symbol']} - {rec['name']} - 相似度: {rec['similarity']:.4f}")
    print()


def test_user_behavior():
    """测试用户行为"""
    print("🖱️ 测试用户行为...")

    response = requests.post(
        f"{BASE_URL}/api/users/behavior/update",
        params={
            "user_id": "test_user",
            "behavior_type": "click",
            "stock_symbol": "AAPL",
            "duration": 30,
            "intensity": 1.0
        }
    )
    print(f"更新用户行为: {response.status_code}")
    if response.status_code == 200:
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def main():
    """主测试函数"""
    print("🚀 开始测试股票推荐系统 API")
    print("=" * 50)

    try:
        test_health()
        time.sleep(1)

        test_user_profile()
        time.sleep(1)

        test_stock_data()
        time.sleep(2)  # 给数据获取一些时间

        test_recommendation()
        time.sleep(1)

        test_user_behavior()
        time.sleep(1)

        test_recommendation()

        print("✅ 所有测试完成！")

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务，请确保服务正在运行")
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")


if __name__ == "__main__":
    main()