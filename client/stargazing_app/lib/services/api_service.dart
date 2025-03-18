import 'dart:convert';
import 'package:http/http.dart' as http;

class StargazingApiService {
  final String baseUrl = 'https://stargazing-hrqk.onrender.com'; // Replace with your API URL
  
  Future<Map<String, dynamic>> predictStargazingQuality(
      double latitude, double longitude, DateTime dateTime) async {
    
    // Format the date as required by API
    final dateFormatted = 
        "${dateTime.year}-${dateTime.month.toString().padLeft(2, '0')}-${dateTime.day.toString().padLeft(2, '0')} "
        "${dateTime.hour.toString().padLeft(2, '0')}:${dateTime.minute.toString().padLeft(2, '0')}:${dateTime.second.toString().padLeft(2, '0')}";
    
    final response = await http.post(
      Uri.parse('$baseUrl/predict/'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'latitude': latitude,
        'longitude': longitude,
        'datetime': dateFormatted
      }),
    );
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to predict stargazing quality: ${response.body}');
    }
  }
}