# generate_report.py
from scanner import run_scan
from datetime import datetime

def create_html_report(scan_results):
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # HTML skeleton and basic CSS styling
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Stock Pattern Scanner</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .card {{ background: white; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; padding: 20px; }}
            .card img {{ max-width: 100%; border-radius: 4px; }}
            .pattern-tag {{ display: inline-block; padding: 5px 10px; background: #007bff; color: white; border-radius: 4px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📈 Post-Market AI Scan Report</h1>
            <p>Scan Date: {today_str}</p>
            <hr>
    """

    # Dynamically generate an HTML card layout for each detected stock
    if not scan_results:
        html_content += "<p>No prominent patterns detected in the market today.</p>"
    else:
        for item in scan_results:
            html_content += f"""
            <div class="card">
                <h2>{item['ticker']}</h2>
                <p><span class="pattern-tag">{item['pattern']}</span> Confidence: {item['confidence']:.2f}%</p>
                <!-- The image saved by the scanner can be loaded directly here -->
                <img src="{item['image_path']}" alt="{item['ticker']} chart">
            </div>
            """

    # Close HTML tags
    html_content += """
        </div>
    </body>
    </html>
    """

    # Write the string to the index.html file
    with open("index.html", "w", encoding="utf-8") as file:
        file.write(html_content)
    print("✅ HTML report (index.html) generated successfully!")

if __name__ == "__main__":
    print("Starting fully automated scanning task...")
    results = run_scan()  # Call the function from scanner.py
    create_html_report(results)