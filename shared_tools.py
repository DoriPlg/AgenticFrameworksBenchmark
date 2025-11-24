from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime, date, timedelta
import sqlite3
import random
import json


# ============= PYDANTIC MODELS =============

class SQLQueryInput(BaseModel):
    """Input model for executing SQL queries"""
    query: str = Field(..., description="SQL SELECT query to execute. Check the tool description for available tables and columns.")


class SQLQueryResult(BaseModel):
    """Result from SQL query execution"""
    success: bool
    rows: List[Dict[str, Any]]
    row_count: int
    error: Optional[str] = None


class WebSearchInput(BaseModel):
    """Input model for web search"""
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum number of results")


class WebSearchResult(BaseModel):
    """Single web search result"""
    title: str
    url: str
    snippet: str
    relevance_score: float


class FlightSearchParams(BaseModel):
    """Structured parameters for flight search (agent should convert to SQL)"""
    origin_airport: str = Field(..., description="Origin airport")
    destination_airport: str = Field(..., description="Destination airport")
    departure_date: date = Field(..., description="Departure date")
    return_date: Optional[date] = Field(default=None, description="Return date, if no return date don't provide this field")
    passengers: int = Field(default=1, ge=1, le=10, description="Number of passengers")
    max_price: Optional[float] = Field(default=None, ge=0, description="Maximum price, if no max price don't provide this field")


class HotelSearchParams(BaseModel):
    """Structured parameters for hotel search (agent should convert to SQL)"""
    city: str
    check_in: date
    check_out: date
    guests: int = 1
    min_rating: Optional[float] = Field(default=None, ge=1, le=5, description="Minimum rating, if no min rating don't provide this field")
    max_price_per_night: Optional[float] = Field(default=None, ge=0, description="Maximum price per night, if no max price per night don't provide this field")


class BookingInput(BaseModel):
    """Input for creating a booking"""
    booking_type: Literal["flight", "hotel","attraction"] = Field(..., description="Type of booking")
    item_id: int = Field(..., description="ID of the item to book")
    customer_name: str = Field(..., description="Name of the customer")
    customer_email: str = Field(..., description="Email of the customer")
    special_requests: Optional[str] = Field(default=None, description="Special requests, if no special requests don't provide this field")


# ============= DATABASE SETUP =============

class TravelDatabase:
    """Mock travel database with real SQL"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialize_database()
    
    def _initialize_database(self):
        """Create tables and populate with mock data (NOW EXPANDED)"""
        cursor = self.conn.cursor()
        
        # Create flights table
        cursor.execute("""
            CREATE TABLE flights (
                flight_id INTEGER PRIMARY KEY AUTOINCREMENT,
                airline TEXT NOT NULL,
                flight_number TEXT NOT NULL,
                origin_airport TEXT NOT NULL,
                destination_airport TEXT NOT NULL,
                departure_date DATE NOT NULL,
                departure_time TIME NOT NULL,
                arrival_time TIME NOT NULL,
                duration_minutes INTEGER NOT NULL,
                base_price REAL NOT NULL,
                cabin_class TEXT NOT NULL,
                available_seats INTEGER NOT NULL,
                aircraft_type TEXT
            )
        """)
        
        # Create hotels table (Added HAS_BAR and HAS_BREAKFAST_INCLUDED)
        cursor.execute("""
            CREATE TABLE hotels (
                hotel_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                address TEXT,
                rating REAL CHECK(rating >= 1 AND rating <= 5),
                price_per_night REAL NOT NULL,
                distance_to_center_km REAL,
                available_rooms INTEGER NOT NULL,
                has_wifi BOOLEAN DEFAULT 1,
                has_pool BOOLEAN DEFAULT 0,
                has_gym BOOLEAN DEFAULT 0,
                has_parking BOOLEAN DEFAULT 0,
                has_spa BOOLEAN DEFAULT 0,
                has_restaurant BOOLEAN DEFAULT 0,
                has_bar BOOLEAN DEFAULT 0,
                has_breakfast_included BOOLEAN DEFAULT 0
            )
        """)
        
        # Create bookings table
        cursor.execute("""
            CREATE TABLE bookings (
                booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
                booking_type TEXT NOT NULL,
                item_id INTEGER NOT NULL,
                customer_name TEXT NOT NULL,
                customer_email TEXT NOT NULL,
                booking_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'confirmed',
                confirmation_number TEXT UNIQUE NOT NULL,
                total_price REAL NOT NULL,
                special_requests TEXT
            )
        """)
        
        # Create attractions table
        cursor.execute("""
            CREATE TABLE attractions (
                attraction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                category TEXT NOT NULL,
                rating REAL,
                description TEXT,
                average_visit_hours REAL,
                entry_fee REAL,
                website TEXT
            )
        """)
        
        # JFK-LAX / LAX-JFK flights with varied dates and times
        flights_data = [
            # Outbound JFK to LAX flights
            ("SkyHigh Airlines", "SH101", "JFK", "LAX", "2025-12-15", "06:00", "09:00", 360, 299.99, "economy", 45, "Boeing 737"),
            ("AirWave", "AW201", "JFK", "LAX", "2025-12-15", "09:30", "12:30", 360, 349.99, "economy", 32, "Airbus A320"),
            ("QuickJet", "QJ301", "JFK", "LAX", "2025-12-15", "12:00", "15:00", 360, 399.99, "economy", 28, "Boeing 737"),
            ("LuxAir", "LX401", "JFK", "LAX", "2025-12-15", "15:30", "18:30", 360, 1299.99, "business", 12, "Boeing 787"),
            ("SkyHigh Airlines", "SH102", "JFK", "LAX", "2025-12-16", "07:00", "10:00", 360, 319.99, "economy", 38, "Boeing 737"),
            ("AirWave", "AW202", "JFK", "LAX", "2025-12-16", "13:00", "16:00", 360, 359.99, "economy", 25, "Airbus A320"),
            ("QuickJet", "QJ302", "JFK", "LAX", "2025-12-17", "10:30", "13:30", 360, 289.99, "economy", 42, "Boeing 737"),
            ("LuxAir", "LX402", "JFK", "LAX", "2025-12-17", "18:00", "21:00", 360, 1199.99, "business", 8, "Boeing 787"),
            
            # Return LAX to JFK flights
            ("SkyHigh Airlines", "SH151", "LAX", "JFK", "2025-12-18", "07:00", "15:20", 380, 279.99, "economy", 52, "Boeing 737"),
            ("AirWave", "AW251", "LAX", "JFK", "2025-12-18", "10:30", "18:50", 380, 329.99, "economy", 35, "Airbus A320"),
            ("QuickJet", "QJ351", "LAX", "JFK", "2025-12-19", "08:00", "16:20", 380, 269.50, "economy", 48, "Boeing 737"),
            ("LuxAir", "LX451", "LAX", "JFK", "2025-12-19", "14:00", "22:20", 380, 1099.99, "business", 10, "Boeing 787"),
            ("SkyHigh Airlines", "SH152", "LAX", "JFK", "2025-12-20", "11:00", "19:20", 380, 299.99, "economy", 40, "Boeing 737"),
            ("AirWave", "AW252", "LAX", "JFK", "2025-12-20", "16:00", "00:20", 380, 349.99, "economy", 30, "Airbus A320"),
            ("QuickJet", "QJ352", "LAX", "JFK", "2025-12-21", "09:00", "17:20", 380, 259.99, "economy", 45, "Boeing 737"),
            ("LuxAir", "LX452", "LAX", "JFK", "2025-12-21", "19:00", "03:20", 380, 1149.99, "business", 6, "Boeing 787"),
            
            # Additional flights to/from other destinations
            # To JFK from other
            ("Delta", "DL1234", "ATL", "JFK", "2025-12-15", "08:00", "10:30", 150, 199.99, "economy", 25, "Boeing 717"),
            ("American", "AA567", "ORD", "JFK", "2025-12-16", "14:00", "16:45", 165, 229.99, "economy", 18, "Airbus A319"),
            
            # From JFK to other
            ("JetBlue", "B6123", "JFK", "BOS", "2025-12-17", "09:00", "09:50", 50, 129.99, "economy", 35, "Embraer 190"),
            ("United", "UA456", "JFK", "SFO", "2025-12-18", "11:00", "14:20", 320, 349.99, "economy", 22, "Boeing 757"),
            
            # From LAX to other
            ("Alaska", "AS789", "LAX", "SEA", "2025-12-19", "10:00", "12:30", 150, 179.99, "economy", 28, "Boeing 737"),
            ("Southwest", "WN321", "LAX", "LAS", "2025-12-20", "16:00", "17:00", 60, 99.99, "economy", 40, "Boeing 737"),
            
            # To LAX from other
            ("Spirit", "NK654", "DFW", "LAX", "2025-12-21", "13:00", "14:15", 135, 159.99, "economy", 45, "Airbus A320"),
            ("Frontier", "F9123", "DEN", "LAX", "2025-12-22", "17:30", "19:00", 150, 139.99, "economy", 30, "Airbus A321"),

            # Cheapest option - connection
            ("SkyHigh Airlines", "SH301", "JFK", "IAH", "2025-12-15", "06:00", "09:00", 360, 29.99, "economy", 45, "Boeing 737"),
            ("SkyHigh Airlines", "SH302", "IAH", "LAX", "2025-12-15", "10:00", "13:00", 360, 29.99, "economy", 45, "Boeing 737"),
        ]
        cursor.executemany("""
            INSERT INTO flights (airline, flight_number, origin_airport, destination_airport, departure_date, 
                               departure_time, arrival_time, duration_minutes, base_price, 
                               cabin_class, available_seats, aircraft_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, flights_data)
        
        # Populate hotels (EXPANDED DATA)
        hotels_data = [
            # Los Angeles (LAX) - (name, city, address, rating, price, dist, rooms, wifi, pool, gym, parking, spa, restaurant, bar, breakfast_included)
            ("Beverly Hills Grand", "Los Angeles", "1 Rodeo Dr", 4.8, 499.99, 0.5, 10, 1, 1, 1, 1, 1, 1, 1, 1),
            ("Santa Monica Beach Hotel", "Los Angeles", "1 Ocean Ave", 4.7, 379.99, 2.0, 15, 1, 1, 1, 1, 1, 1, 1, 1),
            ("Hollywood Plaza", "Los Angeles", "7000 Hollywood Blvd", 4.3, 229.99, 3.5, 20, 1, 1, 1, 1, 0, 1, 1, 0),
            ("LAX Gateway Inn", "Los Angeles", "100 Airport Blvd", 4.0, 159.99, 1.2, 25, 1, 1, 1, 1, 0, 1, 1, 1),
            ("Downtown LA Lofts", "Los Angeles", "800 S Figueroa St", 4.5, 289.99, 1.8, 12, 1, 0, 1, 1, 0, 1, 1, 0),
            ("Venice Beach Hostel", "Los Angeles", "25 Windward Ave", 3.9, 79.99, 4.2, 40, 1, 0, 0, 1, 0, 0, 0, 1),
            ("The Standard, DTLA", "Los Angeles", "550 S Flower St", 4.4, 249.99, 2.1, 18, 1, 1, 1, 1, 0, 1, 1, 0),
            
            # New York (JFK/NYC)
            ("The Plaza", "New York", "768 5th Ave", 4.8, 699.99, 0.1, 5, 1, 1, 1, 1, 1, 1, 1, 1),
            ("Times Square Suites", "New York", "1568 Broadway", 4.2, 349.99, 0.2, 12, 1, 0, 1, 0, 0, 1, 1, 0),
            ("JFK Airport Hotel", "New York", "140-10 20th Ave", 3.8, 179.99, 2.5, 30, 1, 1, 1, 1, 0, 1, 1, 0),
            
            # Other cities
            ("The Drake Chicago", "Chicago", "140 E Walton Pl", 4.6, 279.99, 1.2, 15, 1, 1, 1, 1, 1, 1, 1, 0),
            ("Seattle Waterfront Inn", "Seattle", "2411 Alaskan Way", 4.4, 229.99, 0.3, 20, 1, 0, 1, 1, 0, 1, 1, 1),
            ("The Fairmont San Francisco", "San Francisco", "950 Mason St", 4.7, 399.99, 0.8, 8, 1, 1, 1, 1, 1, 1, 1, 1),
        ]
        cursor.executemany("""
            INSERT INTO hotels (name, city, address, rating, price_per_night, distance_to_center_km,
                              available_rooms, has_wifi, has_pool, has_gym, has_parking, has_spa, 
                              has_restaurant, has_bar, has_breakfast_included)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, hotels_data)
        
        # Populate attractions (EXPANDED DATA)
        attractions_data = [
            # Los Angeles
            ("Hollywood Sign", "Los Angeles", "landmark", 4.7, "Iconic sign overlooking LA", 1.5, None, "hollywood.com"),
            ("Getty Center", "Los Angeles", "museum", 4.8, "Art museum with stunning architecture", 3.0, None, "getty.edu"),
            ("Universal Studios", "Los Angeles", "entertainment", 4.6, "Theme park and film studio", 8.0, 109.00, "universalstudios.com"),
            ("Santa Monica Pier", "Los Angeles", "landmark", 4.5, "Historic pier with amusement park", 2.5, None, "santamonicapier.org"),
            ("Griffith Observatory", "Los Angeles", "museum", 4.8, "Observatory with planetarium", 2.0, None, "griffithobservatory.org"),
            
            # Miami
            ("Vizcaya Museum & Gardens", "Miami", "historic site", 4.7, "European-style mansion and gardens", 3.0, 25.00, "vizcaya.org"),
            ("Everglades National Park", "Miami", "nature/park", 4.6, "Vast wetland ecosystem", 4.0, 30.00, "nps.gov/ever"),
            ("Art Deco Historic District", "Miami", "historic site", 4.5, "Colorful buildings in South Beach", 1.0, None, "artdeco.org"),
            
            # New York
            ("Statue of Liberty", "New York", "landmark", 4.7, "Iconic national monument", 4.0, 24.50, "nps.gov/stli"),
            ("Metropolitan Museum of Art", "New York", "museum", 4.8, "One of the world's largest art museums", 4.0, 30.00, "metmuseum.org"),
            ("Central Park", "New York", "nature/park", 4.9, "Major urban park", 2.0, None, "centralparknyc.org"),
            ("Hollywood Sign", "Los Angeles", "landmark", 4.7, "Iconic sign overlooking LA", 1.5, None, "hollywood.com"),
            ("Getty Center", "Los Angeles", "museum", 4.8, "Art museum with stunning architecture", 3.0, None, "getty.edu"),
            ("Universal Studios", "Los Angeles", "entertainment", 4.6, "Theme park and film studio", 8.0, 109.00, "universalstudios.com"),
            ("Santa Monica Pier", "Los Angeles", "landmark", 4.5, "Historic pier with amusement park", 2.5, None, "santamonicapier.org"),
            ("Griffith Observatory", "Los Angeles", "museum", 4.8, "Observatory with planetarium", 2.0, None, "griffithobservatory.org"),
            ("Dodger Stadium", "Los Angeles", "entertainment", 4.7, "MLB stadium with great views of LA", 4.5, 65.00, "dodgers.com"),
            ("LA Live", "Los Angeles", "entertainment", 4.6, "Performing arts center", 1.0, None, "lalive.com"),
            ("The Getty Villa", "Los Angeles", "museum", 4.8, "Beautiful art museum in a park", 2.0, None, "getty.edu"),
        ]
        cursor.executemany("""
            INSERT INTO attractions (name, city, category, rating, description, average_visit_hours, entry_fee, website)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, attractions_data)
        
        self.conn.commit()
    
    def execute_query(self, input: SQLQueryInput) -> SQLQueryResult:
        """Execute a SQL query and return results"""
        try:
            # Security: Only allow SELECT queries
            if not input.query.strip().upper().startswith("SELECT"):
                return SQLQueryResult(
                    success=False,
                    rows=[],
                    row_count=0,
                    error="Only SELECT queries are allowed"
                )
            
            cursor = self.conn.cursor()
            # if input.params:
            #     cursor.execute(input.query, input.params)
            # else:
            #     cursor.execute(input.query)
            cursor.execute(input.query)
            rows = cursor.fetchall()
            rows_dict = [dict(row) for row in rows]
            
            return SQLQueryResult(
                success=True,
                rows=rows_dict,
                row_count=len(rows_dict)
            )
        
        except sqlite3.Error as e:
            return SQLQueryResult(
                success=False,
                rows=[],
                row_count=0,
                error=str(e)
            )
    
    def create_booking(self, booking: BookingInput) -> Dict[str, Any]:
        """Create a booking (INSERT operation)"""
        try:
            cursor = self.conn.cursor()
            
            # Get price based on item
            if booking.booking_type == "flight":
                cursor.execute("SELECT base_price FROM flights WHERE flight_id = ?", (booking.item_id,))
            else:
                # Assuming 1 night for simplicity in price calculation for hotel
                cursor.execute("SELECT price_per_night FROM hotels WHERE hotel_id = ?", (booking.item_id,))
            
            result = cursor.fetchone()
            if not result:
                return {"success": False, "error": "Item not found"}
            
            total_price = result[0]
            confirmation = f"CONF-{random.randint(100000, 999999)}"
            
            cursor.execute("""
                INSERT INTO bookings (booking_type, item_id, customer_name, customer_email, 
                                    confirmation_number, total_price, special_requests)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (booking.booking_type, booking.item_id, booking.customer_name, 
                  booking.customer_email, confirmation, total_price, booking.special_requests))
            
            self.conn.commit()
            booking_id = cursor.lastrowid
            
            return {
                "success": True,
                "booking_id": booking_id,
                "confirmation_number": confirmation,
                "total_price": total_price,
                "status": "confirmed"
            }
        
        except sqlite3.Error as e:
            return {"success": False, "error": str(e)}
    
    def close(self):
        self.conn.close()


# ============= MOCK WEB SEARCH =============

MOCK_WEB_DATA = {
    "weather Los Angeles": [
        {"title": "Los Angeles Weather Forecast", "url": "weather.com/la", 
         "snippet": "Sunny, 24°C with 10% chance of rain. Humidity 55%. Perfect weather for outdoor activities."},
        {"title": "LA Weather This Week", "url": "accuweather.com/la",
         "snippet": "Expect clear skies throughout the week with temperatures ranging from 20-26°C."},
    ],
    "things to do Los Angeles": [
        {"title": "Top 10 LA Attractions", "url": "tripadvisor.com/la",
         "snippet": "Visit Hollywood Sign, Getty Center, Universal Studios, and Santa Monica Pier."},
        {"title": "Best Activities in LA", "url": "visitcalifornia.com/la",
         "snippet": "Explore museums, beaches, hiking trails, and world-class dining options."},
    ],
    "flights JFK to LAX": [
        {"title": "Compare Flights JFK-LAX", "url": "kayak.com/flights",
         "snippet": "Find cheap flights from New York JFK to Los Angeles LAX. Prices starting at $275."},
    ],
    "Miami travel tips": [
        {"title": "Best Time to Visit Miami", "url": "miamivisit.com/best-time",
         "snippet": "The dry season (November to April) is the most pleasant time to visit Miami."},
    ],
}


def mock_web_search(search_input: WebSearchInput) -> List[WebSearchResult]:
    """Mock web search returning relevant results"""
    query_lower = search_input.query.lower()
    
    results = []
    
    # Simple check for relevance
    relevant_keys = [key for key in MOCK_WEB_DATA.keys() if any(term in query_lower for term in key.lower().split())]

    if relevant_keys:
        for key in relevant_keys:
            for mock in MOCK_WEB_DATA[key][:search_input.max_results]:
                results.append(WebSearchResult(
                    title=mock["title"],
                    url=mock["url"],
                    snippet=mock["snippet"],
                    relevance_score=random.uniform(0.7, 0.95)
                ))
    
    # If no specific match, return generic results
    if not results:
        results.append(WebSearchResult(
            title=f"Results for: {search_input.query}",
            url="search.com/results",
            snippet=f"Information about {search_input.query} from various sources.",
            relevance_score=0.6
        ))
    
    # Filter to max_results and return
    return results[:search_input.max_results]


# ============= TOOL FUNCTIONS =============

# Initialize global database
db = TravelDatabase()


def query_database(query: str) -> SQLQueryResult:
    """
    Query the travel database for flights, hotels, attractions, and bookings.
    
    Available tables:
    - flights
    - hotels
    - attractions
    - bookings
    
    Write a SELECT query to retrieve the information needed.
    """
    #  and columns:
    # - flights: flight_id, airline, flight_number, origin_airport, destination_airport, 
    # departure_date, departure_time, arrival_time, duration_minutes, base_price, 
    # cabin_class, available_seats, aircraft_type
    
    # - hotels: hotel_id, name, city, address, rating, price_per_night, 
    # distance_to_center_km, available_rooms, has_wifi, has_pool, has_gym, 
    # has_parking, has_spa, has_restaurant, has_bar, has_breakfast_included
    
    # - attractions: attraction_id, name, city, category, rating, description, 
    # average_visit_hours, entry_fee, website
        
    # - bookings: booking_id, booking_type, item_id, customer_name, customer_email, 
    # booking_date, status, confirmation_number, total_price, special_requests
    # example row: {
    #     "flight_id": 1,
    #     "airline": "Delta",
    #     "flight_number": "123",
    #     "origin_airport": "CLT",
    #     "destination_airport": "IAH",
    #     "departure_date": "2024-01-01",
    #     "departure_time": "10:00",
    #     "arrival_time": "12:00",
    #     "duration_minutes": 120,
    #     "base_price": 100,
    #     "cabin_class": "business",
    #     "available_seats": 10,
    #     "aircraft_type": "Boeing 737"
    # }
    # example row: {
    #     "hotel_id": 1,
    #     "name": "Hotel A",
    #     "city": "Los Angeles",
    #     "address": "123 Main St",
    #     "rating": 4.5,
    #     "price_per_night": 150,
    #     "distance_to_center_km": 5,
    #     "available_rooms": 10,
    #     "has_wifi": True,
    #     "has_pool": False,
    #     "has_gym": True,
    #     "has_parking": True,
    #     "has_spa": False,
    #     "has_restaurant": True,
    #     "has_bar": True,
    #     "has_breakfast_included": True
    # }
    # example row: {
    #     "attraction_id": 1,
    #     "name": "Attraction A",
    #     "city": "Los Angeles",
    #     "category": "Museum",
    #     "rating": 4.5,
    #     "description": "A fascinating museum with interactive exhibits.",
    #     "average_visit_hours": 2,
    #     "entry_fee": 10,
    #     "website": "attraction.com"
    # }
    # example row: {
    #     "booking_id": 1,
    #     "booking_type": "flight",
    #     "item_id": 1,
    #     "customer_name": "John Doe",
    #     "customer_email": "john.doe@example.com",
    #     "booking_date": "2024-01-01",
    #     "status": "confirmed",
    #     "confirmation_number": "CONF-123456",
    #     "total_price": 100,
    #     "special_requests": "None"
    # }
    return db.execute_query(SQLQueryInput(query=query))


def web_search(query: str, max_results: int) -> List[WebSearchResult]:
    """
    Search online for information.
     (weather, travel tips, etc.)
    """
    return mock_web_search(WebSearchInput(query=query, max_results=max_results))


def create_booking(
    booking_type: Literal["flight", "hotel", "attraction"],
    item_id: int,
    customer_name: str,
    customer_email: str,
    special_requests: str,
) -> Dict[str, Any]:
    """
    Create a new booking in the database
    args:
        booking_type: "flight", "hotel", or "attraction"
        item_id: id of the item to book, as stored in the database
        customer_name: name of the customer
        customer_email: email of the customer
        special_requests: special requests for the booking
    """
    if booking_type == "attraction":
        return {
            "success": True,
            "booking_id": 0,
            "confirmation_number": "CONF-123456",
            "total_price": 0,
            "status": "confirmed"
        }
    return db.create_booking(BookingInput(
        booking_type=booking_type,
        item_id=item_id,
        customer_name=customer_name,
        customer_email=customer_email,
        special_requests=special_requests,
    ))


if __name__ == "__main__":
    print(web_search("weather Los Angeles", 2))
    print(query_database("SELECT * FROM flights"))