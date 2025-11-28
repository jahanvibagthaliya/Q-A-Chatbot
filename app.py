from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)


ESSAY = """On 12 June 2025, India witnessed one of the most devastating aviation disasters in its recent history when Air India Flight AI-171, a Boeing 787-8 Dreamliner, crashed shortly after takeoff from Sardar Vallabhbhai Patel International Airport, Ahmedabad. The flight was scheduled to travel from Ahmedabad to London Gatwick Airport with 242 people on board, including 230 passengers and 12 crew members. It was a routine early-morning departure that turned catastrophic within seconds, marking the first ever fatal crash involving a Boeing 787 aircraft since the model entered commercial service.

Flight AI-171 began its takeoff roll normally and lifted off the runway at approximately 17:48 IST. However, within just 32 seconds of becoming airborne, the aircraft experienced a sudden and critical loss of thrust from both engines. Unable to maintain altitude, it descended rapidly and crashed into a building belonging to B.J. Medical College, located close to the airport's boundary. The impact caused a massive explosion and fire, spreading debris across the campus and nearby residential structures.

The crash resulted in an overwhelming number of fatalities. Of the 242 people on the aircraft, 241 lost their lives, leaving only one survivor, whose escape has been described by authorities as miraculous. On the ground, the crash caused further destruction: 19 people in nearby buildings were killed, and 67 civilians were injured, many with severe burns or trauma from collapsing structures. With a total death count of around 260, the tragedy ranks among the most severe aviation disasters on Indian soil.

The victims included 169 Indian nationals, 53 British nationals, 7 Portuguese citizens, and 1 Canadian passenger. The diversity of the passengers underscored the international impact of the tragedy, with several countries sending teams to assist with victim identification and to support their citizens' families.

The crash triggered an immediate investigation led by India’s Aircraft Accident Investigation Bureau (AAIB), with support from international bodies due to the aircraft’s global relevance. The preliminary investigation report, released several weeks later, revealed a startling and unusual technical situation: the fuel control switches for both engines had moved from the “RUN” position to “CUTOFF” seconds after takeoff. This action effectively starved the engines of fuel, causing them to lose thrust almost simultaneously. The switches were later moved back to “RUN,” but by then at least one engine’s deceleration could not be reversed in time to regain lift.

Cockpit voice recordings captured a brief but intense exchange between the pilots during the emergency, including one pilot urgently asking why the fuel was cut off. The other pilot reportedly denied operating the switches, leaving investigators to probe deeper into the cause. Whether the switch movement resulted from pilot error, an electrical or mechanical malfunction, a design flaw, or a maintenance oversight remains inconclusive based on the preliminary report. The AAIB continues to examine the fuel system, cockpit ergonomics, possible electrical faults, and the aircraft’s maintenance history to determine the final cause.

Firefighters, airport staff, police, and medical teams responded within minutes. However, the fire, fueled by aviation fuel and intensified by the crash into a densely built area, made rescue operations extremely challenging. Many victims’ remains were severely burned or fragmented, necessitating DNA analysis for identification. Hospitals in Ahmedabad, particularly Ahmedabad Civil Hospital, worked tirelessly to conduct testing, facilitate repatriation, and support grieving families. Several nations sent forensic teams to assist in the identification and documentation of their citizens.

In the days following the crash, authorities moved the wreckage to a secure area inside the airport to allow a detailed examination of the engines, control systems, and cockpit components. The tragedy led to nationwide discussions on aviation safety, emergency preparedness, and maintenance protocols. Air India and Indian aviation regulators temporarily increased inspections on other Boeing 787 aircraft operating in the country.

International leaders expressed condolences, and the Indian government announced compensation, long-term support for affected families, and a commitment to transparency in the investigation. The accident has since become a case study in the aviation community due to its unusual circumstances, high fatality count on both air and ground, and its distinction as the first fatal accident involving the advanced Boeing 787 Dreamliner.

Apart from the immense human tragedy, the crash has profound technical and regulatory implications. It raises critical questions about cockpit systems, redundancy, pilot workload, and the design of fuel control mechanisms. The fact that both engines shut down nearly simultaneously makes the event highly unusual, as modern aircraft are designed with multiple safeguards to prevent exactly such a scenario.

The Air India Ahmedabad crash serves as a stark reminder of the complexity and fragility of aviation systems. It has prompted global authorities to evaluate whether there are systemic vulnerabilities in aircraft controls or human-machine interactions that could lead to similar outcomes elsewhere.

The Air India AI-171 crash in Ahmedabad stands as one of the most tragic and puzzling aviation accidents in India’s modern history. With more than 260 lives lost, widespread structural damage, and countless families shattered, the disaster has left an indelible mark on the nation. While the preliminary findings highlight a catastrophic loss of engine thrust caused by unexpected fuel cutoff, the deeper reasons behind that event remain under investigation. The final report, once released, is expected to provide critical insights and shape future aviation safety standards.

This incident not only exposed the vulnerability of even the most technologically advanced aircraft but also emphasized the need for continuous improvements in safety protocols, training, and engineering design. Above all, it stands as a somber reminder of the human cost of aviation failures and the crucial importance of ensuring that such a tragedy never occurs again.
"""


qa_model = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad"
)


@app.route("/")
def home():
    return render_template("index.html", essay=ESSAY)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    result = qa_model({
        "question": question,
        "context": ESSAY
    })

    return jsonify({"answer": result["answer"]})

if __name__ == "__main__":
    app.run(debug=True)



