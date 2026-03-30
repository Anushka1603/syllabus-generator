import io
import json
import re
import sys
from contextlib import redirect_stdout

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL = "llama3.1:8b"
REQUEST_TIMEOUT = 300
NUM_PREDICT = 768

BLOOMS_LEVELS = {
    "Remember": "recall facts and basic concepts (define, list, recall, identify)",
    "Understand": "explain ideas or concepts (describe, explain, summarize, classify)",
    "Apply": "use information in new situations (solve, demonstrate, use, execute)",
    "Analyze": "draw connections among ideas (differentiate, organize, compare, examine)",
    "Evaluate": "justify a decision or course of action (argue, defend, judge, critique)",
    "Create": "produce new or original work (design, construct, develop, formulate)",
}

SECTION_LABELS = {
    "course_objectives": "Course Objectives",
    "course_outcomes": "Course Outcomes",
    "program_objectives": "Program Objectives",
    "program_outcomes": "Program Outcomes",
    "syllabus": "Unit-wise Syllabus",
    "books": "Textbooks and References",
    "youtube": "YouTube Suggestions",
}


def query_ollama(prompt: str, system: str = "") -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": NUM_PREDICT,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        text = result.get("response", "")
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to Ollama.")
        print("Make sure Ollama is running with: ollama serve")
        print(f"Make sure the model is pulled with: ollama pull {MODEL}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR querying Ollama: {exc}")
        return ""


def check_ollama_connection() -> bool:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()
        models = [model["name"] for model in response.json().get("models", [])]
        return any(name.startswith(MODEL.split(":")[0]) for name in models)
    except Exception:
        return False


def parse_json_array(raw: str, limit: int) -> list[str]:
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item).strip() for item in data[:limit] if str(item).strip()]
    except Exception:
        pass

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [str(item).strip() for item in data[:limit] if str(item).strip()]
        except Exception:
            pass

    lines = []
    for line in raw.splitlines():
        cleaned = line.strip().strip('"').strip("'").strip()
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
        cleaned = re.sub(r"^(CO|PO|PEO)\d+[:.-]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.lstrip("-* ").strip()
        if len(cleaned) > 10 and not cleaned.lower().startswith("here are"):
            lines.append(cleaned)

    deduped = []
    for item in lines:
        if item not in deduped:
            deduped.append(item)

    return deduped[:limit]


def parse_json_object(raw: str) -> dict:
    if not raw:
        return {}

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def parse_json_list_of_objects(raw: str) -> list[dict]:
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def generate_course_objectives(program: str, course: str, level: str) -> list[str]:
    system = (
        "You are an expert curriculum designer. Return only plain JSON. "
        "No markdown, no explanation, no headings, no code fences."
    )
    prompt = f"""
Generate exactly 4 course objectives for:
- Program: {program}
- Course/Subject: {course}
- Academic Level: {level}

Rules:
- each objective must start with a strong Bloom's Taxonomy action verb
- each objective must be one sentence only
- keep each objective under 20 words
- do not number the objectives
- do not add labels like CO1

Return only this JSON format:
["Objective 1", "Objective 2", "Objective 3", "Objective 4"]
"""
    return parse_json_array(
        query_ollama(prompt, system),
        limit=5,
    )


def generate_course_outcomes(program: str, course: str, level: str) -> list[str]:
    system = (
        "You are an expert curriculum designer. Respond only with a JSON array of strings. "
        "Do not include markdown fences or extra commentary."
    )
    prompt = f"""
Generate 4 to 5 course outcomes for:
- Program: {program}
- Course/Subject: {course}
- Academic Level: {level}

Each outcome must:
- start with 'After completing this course, students will be able to...'
- use measurable Bloom's Taxonomy verbs
- include a Bloom's level tag at the end, for example [Bloom's: Apply]
- be distinct and concise

Return only a JSON array like ["CO1: ...", "CO2: ..."]
"""
    return parse_json_array(
        query_ollama(prompt, system),
        limit=5,
    )


def generate_program_objectives(program: str, level: str) -> list[str]:
    system = (
        "You are an expert curriculum designer. Respond only with a JSON array of strings. "
        "Do not include markdown fences or extra commentary."
    )
    prompt = f"""
Generate 4 to 5 program educational objectives for:
- Program: {program}
- Academic Level: {level}

Each objective must:
- be broad and reflect the overall mission of the program
- address professional skills, ethics, and lifelong learning when applicable
- stay concise

Return only a JSON array like ["PO1: ...", "PO2: ..."]
"""
    return parse_json_array(
        query_ollama(prompt, system),
        limit=5,
    )


def generate_program_outcomes(program: str, level: str) -> list[str]:
    system = (
        "You are an expert curriculum designer. Respond only with a JSON array of strings. "
        "Do not include markdown fences or extra commentary."
    )
    prompt = f"""
Generate 5 to 6 program outcomes for:
- Program: {program}
- Academic Level: {level}

Each outcome must:
- describe what graduates will know or be able to do
- include a Bloom's level tag at the end, for example [Bloom's: Evaluate]
- cover technical competence, communication, ethics, and teamwork
- stay concise

Return only a JSON array like ["PO1: ...", "PO2: ..."]
"""
    return parse_json_array(
        query_ollama(prompt, system),
        limit=6,
    )


def generate_syllabus(program: str, course: str, level: str, num_units: int) -> dict:
    system = (
        "You are an expert curriculum designer. Respond only with valid JSON. "
        "Do not include markdown fences or extra commentary."
    )
    prompt = f"""
Generate a concise unit-wise syllabus for:
- Program: {program}
- Course/Subject: {course}
- Academic Level: {level}
- Number of Units: {num_units}

For each unit provide:
- unit_number: integer
- unit_title: string
- duration_hours: integer
- bloom_level: one of [Remember, Understand, Apply, Analyze, Evaluate, Create]
- topics: array of 3 to 4 topic strings
- subtopics: object where keys are topic names and values are arrays of 1 to 2 subtopic strings
- learning_outcomes: array of 2 outcome strings using Bloom's verbs

Return only a JSON object in the form {{"units": [...]}}.
"""
    return parse_json_object(query_ollama(prompt, system))


def generate_textbooks(program: str, course: str, level: str) -> dict:
    system = (
        "You are an expert curriculum designer and librarian. Respond only with valid JSON. "
        "Do not include markdown fences or extra commentary."
    )
    prompt = f"""
Recommend textbooks and reference books for:
- Program: {program}
- Course/Subject: {course}
- Academic Level: {level}

Provide:
- 2 to 3 main textbooks
- 2 to 3 reference books

For each book include:
- title
- authors
- publisher
- year
- edition
- reason

Return only a JSON object with keys textbooks and reference_books.
"""
    return parse_json_object(query_ollama(prompt, system))


def generate_youtube_suggestions(program: str, course: str, level: str) -> list[dict]:
    system = (
        "You are an expert educator and content curator. Respond only with a JSON array. "
        "Do not include markdown fences or extra commentary."
    )
    prompt = f"""
Suggest 4 to 5 YouTube channels, playlists, or video series for learning:
- Program: {program}
- Course/Subject: {course}
- Academic Level: {level}

For each suggestion include:
- channel_or_playlist
- description
- bloom_level
- type: one of [Channel, Playlist, Video Series]
- search_query

Return only a JSON array.
"""
    return parse_json_list_of_objects(query_ollama(prompt, system))


def print_rule(char: str = "=", width: int = 72) -> None:
    print(char * width)


def print_header(title: str) -> None:
    print()
    print_rule("=")
    print(title)
    print_rule("=")


def print_section(title: str) -> None:
    print()
    print_rule("-", 60)
    print(title)
    print_rule("-", 60)


def display_header(info: dict) -> None:
    print_rule("#")
    print("AI SYLLABUS GENERATOR")
    print_rule("#")
    print(f"Program : {info['program']}")
    print(f"Course  : {info['course']}")
    print(f"Level   : {info['level']}")
    print(f"Units   : {info['num_units']}")


def display_section(section_key: str, results: dict) -> None:
    if section_key == "course_objectives":
        print_header("COURSE OBJECTIVES")
        items = results.get(section_key, [])
        if not items:
            print("No valid course objectives were returned.")
        for index, objective in enumerate(items, start=1):
            print(f"CO{index}. {objective}")
    elif section_key == "course_outcomes":
        print_header("COURSE OUTCOMES")
        items = results.get(section_key, [])
        if not items:
            print("No valid course outcomes were returned.")
        for outcome in items:
            print(f"- {outcome}")
    elif section_key == "program_objectives":
        print_header("PROGRAM OBJECTIVES")
        items = results.get(section_key, [])
        if not items:
            print("No valid program objectives were returned.")
        for index, objective in enumerate(items, start=1):
            print(f"PEO{index}. {objective}")
    elif section_key == "program_outcomes":
        print_header("PROGRAM OUTCOMES")
        items = results.get(section_key, [])
        if not items:
            print("No valid program outcomes were returned.")
        for outcome in items:
            print(f"- {outcome}")
    elif section_key == "syllabus":
        print_header("UNIT-WISE SYLLABUS")
        units = results.get(section_key, {}).get("units", [])
        if not units:
            print("No valid syllabus units were returned.")
        for unit in units:
            print_rule("-", 60)
            print(f"UNIT {unit.get('unit_number', '?')}: {unit.get('unit_title', '')}")
            print(f"Duration Hours : {unit.get('duration_hours', '?')}")
            print(f"Bloom Level    : {unit.get('bloom_level', '?')}")
            print("Topics:")
            for topic in unit.get("topics", []):
                print(f"- {topic}")
                for subtopic in unit.get("subtopics", {}).get(topic, []):
                    print(f"  * {subtopic}")
            print("Learning Outcomes:")
            for outcome in unit.get("learning_outcomes", []):
                print(f"- {outcome}")
    elif section_key == "books":
        books = results.get(section_key, {})
        print_header("TEXTBOOKS")
        textbooks = books.get("textbooks", [])
        reference_books = books.get("reference_books", [])
        if not textbooks and not reference_books:
            print("No valid book recommendations were returned.")
        for index, book in enumerate(textbooks, start=1):
            print(f"[{index}] {book.get('title', 'N/A')}")
            print(f"  Authors   : {book.get('authors', 'N/A')}")
            print(f"  Publisher : {book.get('publisher', 'N/A')}")
            print(f"  Year      : {book.get('year', 'N/A')}")
            print(f"  Edition   : {book.get('edition', 'N/A')}")
            print(f"  Reason    : {book.get('reason', 'N/A')}")
        print_section("REFERENCE BOOKS")
        for index, book in enumerate(reference_books, start=1):
            print(f"[{index}] {book.get('title', 'N/A')}")
            print(f"  Authors   : {book.get('authors', 'N/A')}")
            print(f"  Publisher : {book.get('publisher', 'N/A')}")
            print(f"  Year      : {book.get('year', 'N/A')}")
            print(f"  Edition   : {book.get('edition', 'N/A')}")
            print(f"  Reason    : {book.get('reason', 'N/A')}")
    elif section_key == "youtube":
        print_header("YOUTUBE SUGGESTIONS")
        items = results.get(section_key, [])
        if not items:
            print("No valid YouTube suggestions were returned.")
        for index, item in enumerate(items, start=1):
            print(f"[{index}] {item.get('channel_or_playlist', 'N/A')} ({item.get('type', 'Channel')})")
            print(f"  Bloom Level : {item.get('bloom_level', 'N/A')}")
            print(f"  Description : {item.get('description', 'N/A')}")
            print(f"  Search Query: {item.get('search_query', 'N/A')}")


def display_results(data: dict) -> None:
    display_header(data["info"])
    print_header("BLOOM'S TAXONOMY REFERENCE")
    for level_name, description in BLOOMS_LEVELS.items():
        print(f"- {level_name:<10} : {description}")
    for section_key in SECTION_LABELS:
        if section_key in data:
            display_section(section_key, data)
    print()
    print_rule("#")
    print("GENERATION COMPLETE")
    print_rule("#")


def save_to_file(data: dict, filename: str) -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        display_results(data)

    with open(filename, "w", encoding="utf-8") as file_handle:
        file_handle.write(buffer.getvalue())

    print(f"Output saved to: {filename}")


def get_user_input() -> dict:
    print()
    print_rule("=", 60)
    print("SYLLABUS GENERATOR - Powered by qwen3:4b")
    print_rule("=", 60)

    program_levels = {
        "1": "Diploma",
        "2": "Undergraduate (B.Tech/B.E.)",
        "3": "Undergraduate (B.Sc.)",
        "4": "Undergraduate (BCA/BCS)",
        "5": "Undergraduate (B.Com/BBA)",
        "6": "Postgraduate (M.Tech/M.E.)",
        "7": "Postgraduate (M.Sc.)",
        "8": "Postgraduate (MCA/MBA)",
        "9": "PhD / Doctoral",
        "10": "Other (Custom)",
    }

    print("\nSelect Academic Level:")
    for key, value in program_levels.items():
        print(f"{key}. {value}")

    while True:
        choice = input("\nEnter choice (1-10): ").strip()
        if choice in program_levels:
            level = program_levels[choice]
            if choice == "10":
                level = input("Enter custom level: ").strip()
            break
        print("Invalid choice. Try again.")

    program = input("\nEnter Program Name (for example, B.Tech Computer Science): ").strip()
    if not program:
        program = f"{level} Program"

    course = input("Enter Course/Subject Name (for example, Data Structures): ").strip()
    if not course:
        course = "Core Subject"

    while True:
        try:
            num_units = int(input("Number of Units/Modules (1-10): ").strip())
            if 1 <= num_units <= 10:
                break
            print("Enter a number between 1 and 10.")
        except ValueError:
            print("Please enter a valid integer.")

    save = input("\nSave results to file after each generated section? (y/n): ").strip().lower() == "y"

    return {
        "level": level,
        "program": program,
        "course": course,
        "num_units": num_units,
        "save": save,
    }


def get_output_filename(program: str, course: str) -> str:
    safe_name = re.sub(r"[^\w\s-]", "", f"{course}_{program}").replace(" ", "_")
    return f"syllabus_{safe_name}.txt"


def build_generation_actions(program: str, course: str, level: str, num_units: int) -> dict:
    return {
        "course_objectives": lambda: generate_course_objectives(program, course, level),
        "course_outcomes": lambda: generate_course_outcomes(program, course, level),
        "program_objectives": lambda: generate_program_objectives(program, level),
        "program_outcomes": lambda: generate_program_outcomes(program, level),
        "syllabus": lambda: generate_syllabus(program, course, level, num_units),
        "books": lambda: generate_textbooks(program, course, level),
        "youtube": lambda: generate_youtube_suggestions(program, course, level),
    }


def print_menu(results: dict) -> None:
    print()
    print_rule("-", 60)
    print("Select a section to generate")
    print_rule("-", 60)
    for index, (key, label) in enumerate(SECTION_LABELS.items(), start=1):
        status = "done" if key in results else "pending"
        print(f"{index}. {label} [{status}]")
    print("8. Generate all remaining sections")
    print("9. Show generated sections so far")
    print("0. Exit")


def main() -> None:
    print("\nChecking Ollama connection...", end="", flush=True)
    if not check_ollama_connection():
        print(f"\n\nWARNING: Could not verify '{MODEL}' in Ollama.")
        print("Ensure Ollama is running and the model is pulled:")
        print("  ollama serve")
        print(f"  ollama pull {MODEL}")
        continue_anyway = input("\nContinue anyway? (y/n): ").strip().lower()
        if continue_anyway != "y":
            sys.exit(0)
    else:
        print(" connected.")

    user_input = get_user_input()
    results = {
        "info": {
            "program": user_input["program"],
            "course": user_input["course"],
            "level": user_input["level"],
            "num_units": user_input["num_units"],
        }
    }

    actions = build_generation_actions(
        user_input["program"],
        user_input["course"],
        user_input["level"],
        user_input["num_units"],
    )
    output_filename = get_output_filename(user_input["program"], user_input["course"])

    while True:
        print_menu(results)
        choice = input("\nEnter your choice: ").strip()

        if choice == "0":
            print("Exiting.")
            break

        if choice == "9":
            display_results(results)
            continue

        if choice == "8":
            pending_keys = [key for key in SECTION_LABELS if key not in results]
            if not pending_keys:
                print("All sections have already been generated.")
                continue
            for key in pending_keys:
                print(f"\nGenerating {SECTION_LABELS[key]}...", end="", flush=True)
                results[key] = actions[key]()
                print(" done")
                display_header(results["info"])
                display_section(key, results)
                if user_input["save"]:
                    save_to_file(results, output_filename)
            continue

        if choice in {str(index) for index in range(1, 8)}:
            section_key = list(SECTION_LABELS.keys())[int(choice) - 1]
            print(f"\nGenerating {SECTION_LABELS[section_key]}...", end="", flush=True)
            results[section_key] = actions[section_key]()
            print(" done")
            display_header(results["info"])
            display_section(section_key, results)
            if user_input["save"]:
                save_to_file(results, output_filename)
            continue

        print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
