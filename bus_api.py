from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncpg
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class OccupancyData(BaseModel):
    camera_id: str
    occupancy: int
    capacity: int
    timestamp: Optional[datetime] = None

class AlertData(BaseModel):
    camera_id: str
    alert_type: str
    message: str
    occupancy: int
    capacity: int

# Database connection pool
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage database connection pool lifecycle"""
    global db_pool
    # Startup
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is required")
    
    # Convert postgres:// to postgresql:// if needed (for Neon compatibility)
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    try:
        db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
            max_inactive_connection_lifetime=300,
            command_timeout=60
        )
        logger.info("Database pool created successfully")
    except Exception as e:
        logger.error(f"Failed to create database pool: {e}")
        raise
    
    yield
    
    # Shutdown
    if db_pool:
        await db_pool.close()
        logger.info("Database pool closed")

# Initialize FastAPI app
app = FastAPI(
    title="Bus Occupancy API",
    description="API for tracking bus occupancy data with PostgreSQL",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Database Operations
# ========================
async def ensure_bus_exists(camera_id: str):
    """Ensure bus exists in database"""
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO buses (camera_id, bus_number, route)
            VALUES ($1, $2, $3)
            ON CONFLICT (camera_id) DO NOTHING
        """, camera_id, f"BUS-{camera_id[-3:]}", "Unknown Route")

async def save_occupancy(data: OccupancyData):
    """Save occupancy data to database"""
    async with db_pool.acquire() as conn:
        await ensure_bus_exists(data.camera_id)
        
        await conn.execute("""
            INSERT INTO occupancy_logs (camera_id, occupancy, capacity, timestamp)
            VALUES ($1, $2, $3, $4)
        """, data.camera_id, data.occupancy, data.capacity, 
            data.timestamp or datetime.utcnow())

async def save_alert(alert: AlertData):
    """Save alert to database"""
    async with db_pool.acquire() as conn:
        await ensure_bus_exists(alert.camera_id)
        
        await conn.execute("""
            INSERT INTO alerts (camera_id, alert_type, message, occupancy, capacity)
            VALUES ($1, $2, $3, $4, $5)
        """, alert.camera_id, alert.alert_type, alert.message, 
            alert.occupancy, alert.capacity)

# ========================
# API Endpoints
# ========================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Bus Occupancy API",
        "status": "running",
        "database": "connected" if db_pool else "disconnected",
        "endpoints": {
            "/api/occupancy": "POST - Submit occupancy data",
            "/api/occupancy/{camera_id}": "GET - Get latest occupancy",
            "/api/history/{camera_id}": "GET - Get occupancy history",
            "/api/stats": "GET - Get system statistics",
            "/api/buses": "GET - Get all buses",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed")

@app.post("/api/occupancy")
async def submit_occupancy(data: OccupancyData):
    """Submit occupancy data"""
    try:
        # Save to database
        await save_occupancy(data)
        
        # Check if alert needed
        if data.occupancy > data.capacity:
            alert = AlertData(
                camera_id=data.camera_id,
                alert_type="OVERCAPACITY",
                message=f"Bus {data.camera_id} is over capacity: {data.occupancy}/{data.capacity}",
                occupancy=data.occupancy,
                capacity=data.capacity
            )
            await save_alert(alert)
            logger.warning(f"Overcapacity alert for {data.camera_id}")
        
        elif data.occupancy >= data.capacity * 0.9:
            alert = AlertData(
                camera_id=data.camera_id,
                alert_type="NEAR_FULL",
                message=f"Bus {data.camera_id} is near full: {data.occupancy}/{data.capacity}",
                occupancy=data.occupancy,
                capacity=data.capacity
            )
            await save_alert(alert)
        
        return {
            "status": "success",
            "camera_id": data.camera_id,
            "occupancy": data.occupancy,
            "capacity": data.capacity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to submit occupancy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/occupancy/{camera_id}")
async def get_latest_occupancy(camera_id: str):
    """Get latest occupancy for a specific camera"""
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT occupancy, capacity, timestamp
                FROM occupancy_logs
                WHERE camera_id = $1
                ORDER BY timestamp DESC
                LIMIT 1
            """, camera_id)
            
            if not result:
                raise HTTPException(status_code=404, detail="No data found for this camera")
            
            return {
                "camera_id": camera_id,
                "occupancy": result["occupancy"],
                "capacity": result["capacity"],
                "timestamp": result["timestamp"].isoformat(),
                "status": "FULL" if result["occupancy"] >= result["capacity"] else 
                         "NEAR_FULL" if result["occupancy"] >= result["capacity"] * 0.8 else "OK"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get occupancy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{camera_id}")
async def get_occupancy_history(
    camera_id: str,
    hours: int = 24,
    limit: int = 100
):
    """Get occupancy history for a specific camera"""
    try:
        async with db_pool.acquire() as conn:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            results = await conn.fetch("""
                SELECT occupancy, capacity, timestamp
                FROM occupancy_logs
                WHERE camera_id = $1 AND timestamp > $2
                ORDER BY timestamp DESC
                LIMIT $3
            """, camera_id, since, limit)
            
            return {
                "camera_id": camera_id,
                "period_hours": hours,
                "count": len(results),
                "history": [
                    {
                        "occupancy": row["occupancy"],
                        "capacity": row["capacity"],
                        "timestamp": row["timestamp"].isoformat()
                    }
                    for row in results
                ]
            }
            
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_system_stats():
    """Get system-wide statistics"""
    try:
        async with db_pool.acquire() as conn:
            # Get active buses (reported in last hour)
            active_buses = await conn.fetchval("""
                SELECT COUNT(DISTINCT camera_id)
                FROM occupancy_logs
                WHERE timestamp > NOW() - INTERVAL '1 hour'
            """)
            
            # Get total records today
            total_today = await conn.fetchval("""
                SELECT COUNT(*)
                FROM occupancy_logs
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
            
            # Get alerts today
            alerts_today = await conn.fetchval("""
                SELECT COUNT(*)
                FROM alerts
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            
            # Get current occupancy stats
            current_stats = await conn.fetch("""
                WITH latest_occupancy AS (
                    SELECT DISTINCT ON (camera_id)
                        camera_id,
                        occupancy,
                        capacity,
                        timestamp
                    FROM occupancy_logs
                    WHERE timestamp > NOW() - INTERVAL '30 minutes'
                    ORDER BY camera_id, timestamp DESC
                )
                SELECT 
                    COUNT(*) as total_buses,
                    SUM(occupancy) as total_occupancy,
                    SUM(capacity) as total_capacity,
                    AVG(occupancy::float / NULLIF(capacity, 0) * 100) as avg_occupancy_percent
                FROM latest_occupancy
            """)
            
            stats = current_stats[0] if current_stats else {}
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "active_buses": active_buses,
                "total_records_24h": total_today,
                "alerts_24h": alerts_today,
                "current_stats": {
                    "total_buses": stats.get("total_buses", 0),
                    "total_occupancy": stats.get("total_occupancy", 0),
                    "total_capacity": stats.get("total_capacity", 0),
                    "average_occupancy_percent": round(stats.get("avg_occupancy_percent", 0) or 0, 2)
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/buses")
async def get_all_buses():
    """Get all buses with their latest status"""
    try:
        async with db_pool.acquire() as conn:
            results = await conn.fetch("""
                WITH latest_occupancy AS (
                    SELECT DISTINCT ON (b.camera_id)
                        b.camera_id,
                        b.bus_number,
                        b.route,
                        o.occupancy,
                        o.capacity,
                        o.timestamp
                    FROM buses b
                    LEFT JOIN occupancy_logs o ON b.camera_id = o.camera_id
                    ORDER BY b.camera_id, o.timestamp DESC
                )
                SELECT * FROM latest_occupancy
                ORDER BY camera_id
            """)
            
            buses = []
            for row in results:
                status = "UNKNOWN"
                if row["occupancy"] is not None and row["capacity"] is not None:
                    if row["occupancy"] >= row["capacity"]:
                        status = "FULL"
                    elif row["occupancy"] >= row["capacity"] * 0.8:
                        status = "NEAR_FULL"
                    else:
                        status = "OK"
                
                buses.append({
                    "camera_id": row["camera_id"],
                    "bus_number": row["bus_number"],
                    "route": row["route"],
                    "occupancy": row["occupancy"],
                    "capacity": row["capacity"],
                    "status": status,
                    "last_update": row["timestamp"].isoformat() if row["timestamp"] else None
                })
            
            return {
                "count": len(buses),
                "buses": buses
            }
            
    except Exception as e:
        logger.error(f"Failed to get buses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts/{camera_id}")
async def get_bus_alerts(camera_id: str, limit: int = 50):
    """Get alerts for a specific bus"""
    try:
        async with db_pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT alert_type, message, occupancy, capacity, created_at
                FROM alerts
                WHERE camera_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, camera_id, limit)
            
            return {
                "camera_id": camera_id,
                "count": len(results),
                "alerts": [
                    {
                        "type": row["alert_type"],
                        "message": row["message"],
                        "occupancy": row["occupancy"],
                        "capacity": row["capacity"],
                        "timestamp": row["created_at"].isoformat()
                    }
                    for row in results
                ]
            }
            
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)