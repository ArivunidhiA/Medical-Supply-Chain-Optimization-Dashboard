import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def generate_supply_chain_data():
    """Generate realistic medical supply chain data"""
    np.random.seed(42)
    
    # Generate dates for one year
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    products = ['Surgical Masks', 'Medical Gloves', 'Syringes', 'Bandages', 'Sanitizers']
    suppliers = ['MedSupply Co', 'HealthEquip Inc', 'MediCore Ltd']
    
    # Generate base data
    data = []
    for date in dates:
        for product in products:
            # Base demand with seasonal pattern
            base_demand = 1000 + 500 * np.sin(date.month * np.pi / 6)
            
            # Add random variation
            demand = int(np.random.normal(base_demand, base_demand * 0.1))
            
            # Generate inventory levels
            inventory = int(np.random.normal(base_demand * 1.2, base_demand * 0.15))
            
            # Assign supplier
            supplier = np.random.choice(suppliers)
            
            # Generate delivery times (days)
            delivery_time = int(np.random.normal(3, 1))
            
            # Calculate stockout risk
            stockout_risk = max(0, min(1, (demand - inventory) / demand if demand > 0 else 0))
            
            data.append({
                'Date': date,
                'Product': product,
                'Supplier': supplier,
                'Demand': demand,
                'Inventory': inventory,
                'DeliveryTime': delivery_time,
                'StockoutRisk': stockout_risk
            })
    
    return pd.DataFrame(data)

def create_dashboard(df):
    """Create and save the supply chain dashboard"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Inventory vs Demand Trends', 'Supplier Performance',
                       'Product Stockout Risk', 'Delivery Time Analysis'),
        specs=[[{"secondary_y": True}, {}],
               [{}, {}]]
    )
    
    # 1. Inventory vs Demand Trends
    inventory_demand = df.groupby('Date').agg({
        'Inventory': 'sum',
        'Demand': 'sum'
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(x=inventory_demand['Date'], y=inventory_demand['Inventory'],
                  name="Total Inventory", line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=inventory_demand['Date'], y=inventory_demand['Demand'],
                  name="Total Demand", line=dict(color='red')),
        row=1, col=1
    )
    
    # 2. Supplier Performance
    supplier_perf = df.groupby('Supplier').agg({
        'DeliveryTime': 'mean',
        'StockoutRisk': 'mean'
    }).reset_index()
    
    fig.add_trace(
        go.Bar(x=supplier_perf['Supplier'],
               y=supplier_perf['DeliveryTime'],
               name="Avg Delivery Time"),
        row=1, col=2
    )
    
    # 3. Product Stockout Risk
    stockout_risk = df.groupby('Product')['StockoutRisk'].mean().reset_index()
    
    fig.add_trace(
        go.Bar(x=stockout_risk['Product'],
               y=stockout_risk['StockoutRisk'],
               name="Stockout Risk",
               marker_color='red'),
        row=2, col=1
    )
    
    # 4. Delivery Time Analysis
    delivery_trend = df.groupby('Date')['DeliveryTime'].mean().reset_index()
    
    fig.add_trace(
        go.Scatter(x=delivery_trend['Date'],
                  y=delivery_trend['DeliveryTime'],
                  name="Avg Delivery Time",
                  line=dict(color='green')),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(height=800, width=1200,
                     title_text="Medical Supply Chain Dashboard",
                     showlegend=True)
    fig.update_xaxes(tickangle=45)
    
    # Save dashboard
    fig.write_html("medical_supply_dashboard.html")

def generate_forecasts(df):
    """Generate demand forecasts and insights"""
    # Prepare data for forecasting
    daily_demand = df.groupby('Date')['Demand'].sum().reset_index()
    daily_demand['Days'] = (daily_demand['Date'] - daily_demand['Date'].min()).dt.days
    
    # Create and fit model
    X = daily_demand['Days'].values.reshape(-1, 1)
    y = daily_demand['Demand'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions for next 30 days
    future_days = np.array(range(len(X), len(X) + 30)).reshape(-1, 1)
    future_demand = model.predict(future_days)
    
    # Save insights
    with open("supply_chain_insights.txt", "w") as f:
        f.write("Medical Supply Chain Insights:\n\n")
        
        # Overall metrics
        f.write(f"Average Daily Demand: {df['Demand'].mean():,.0f} units\n")
        f.write(f"Average Inventory Level: {df['Inventory'].mean():,.0f} units\n")
        f.write(f"Average Delivery Time: {df['DeliveryTime'].mean():.1f} days\n")
        
        # Risk analysis
        high_risk_products = df.groupby('Product')['StockoutRisk'].mean()
        f.write(f"\nHigh-Risk Products (Stockout Risk > 0.3):\n")
        for product, risk in high_risk_products[high_risk_products > 0.3].items():
            f.write(f"- {product}: {risk:.1%} risk\n")
        
        # Supplier performance
        supplier_perf = df.groupby('Supplier')['DeliveryTime'].mean()
        f.write(f"\nSupplier Performance:\n")
        for supplier, time in supplier_perf.items():
            f.write(f"- {supplier}: {time:.1f} days avg delivery\n")
        
        # Forecast
        f.write(f"\nDemand Forecast:\n")
        f.write(f"Projected 30-day demand trend: ")
        if model.coef_[0] > 0:
            f.write("Increasing\n")
        else:
            f.write("Decreasing\n")
        f.write(f"Average projected daily demand: {future_demand.mean():,.0f} units\n")

def main():
    # Generate data
    print("Generating supply chain data...")
    df = generate_supply_chain_data()
    
    # Create dashboard
    print("Creating dashboard...")
    create_dashboard(df)
    
    # Generate forecasts and insights
    print("Generating forecasts and insights...")
    generate_forecasts(df)
    
    print("\nDashboard and insights have been generated successfully!")
    print("Open 'medical_supply_dashboard.html' in your web browser to view the dashboard.")
    print("Check 'supply_chain_insights.txt' for detailed analysis and forecasts.")

if __name__ == "__main__":
    main()
