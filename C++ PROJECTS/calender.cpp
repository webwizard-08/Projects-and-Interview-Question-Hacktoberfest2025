#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
using namespace std;

struct Event {
    int day, month, year;
    string description;
};

vector<Event> events;

bool isLeapYear(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

int getDaysInMonth(int month, int year) {
    int daysInMonth[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2 && isLeapYear(year)) return 29;
    return daysInMonth[month - 1];
}

int getStartDay(int month, int year) {
    int y = year;
    int m = month;
    if (m < 3) {
        y--;
        m += 12;
    }
    int K = y % 100;
    int J = y / 100;
    return (1 + (13 * (m + 1)) / 5 + K + (K / 4) + (J / 4) + (5 * J)) % 7;
}

void displayCalendar(int month, int year) {
    string months[] = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"};
    cout << "\n  " << months[month - 1] << " " << year << "\n";
    cout << "  Sun Mon Tue Wed Thu Fri Sat\n";
    
    int startDay = getStartDay(month, year);
    int days = getDaysInMonth(month, year);

    for (int i = 0; i < startDay; i++) cout << "    ";
    for (int day = 1; day <= days; day++) {
        cout << setw(4) << day;
        if ((startDay + day) % 7 == 0) cout << endl;
    }
    cout << "\n";
}

void saveEvents() {
    ofstream file("events.txt");
    for (const auto &event : events)
        file << event.day << " " << event.month << " " << event.year << " " << event.description << endl;
    file.close();
}

void loadEvents() {
    ifstream file("events.txt");
    Event e;
    while (file >> e.day >> e.month >> e.year) {
        file.ignore();
        getline(file, e.description);
        events.push_back(e);
    }
    file.close();
}

void addEvent() {
    Event e;
    cout << "Enter date (DD MM YYYY): ";
    cin >> e.day >> e.month >> e.year;
    cin.ignore();
    cout << "Enter event description: ";
    getline(cin, e.description);
    events.push_back(e);
    saveEvents();
    cout << "Event added successfully!\n";
}

void viewEvents() {
    int month, year;
    cout << "Enter month and year: ";
    cin >> month >> year;
    cout << "\nEvents for " << month << "/" << year << ":\n";
    for (const auto &e : events)
        if (e.month == month && e.year == year)
            cout << e.day << ": " << e.description << endl;
}

void deleteEvent() {
    int day, month, year;
    cout << "Enter date to delete event (DD MM YYYY): ";
    cin >> day >> month >> year;
    for (auto it = events.begin(); it != events.end(); ++it) {
        if (it->day == day && it->month == month && it->year == year) {
            events.erase(it);
            saveEvents();
            cout << "Event deleted successfully!\n";
            return;
        }
    }
    cout << "Event not found!\n";
}

int main() {
    loadEvents();
    int choice, month, year;
    do {
        cout << "\nCalendar Application\n";
        cout << "1. View Calendar\n2. Add Event\n3. View Events\n4. Delete Event\n5. Exit\nEnter choice: ";
        cin >> choice;
        switch (choice) {
            case 1:
                cout << "Enter month and year: ";
                cin >> month >> year;
                if (month < 1 || month > 12 || year < 1) {
                    cout << "Invalid input! Please enter a valid month (1-12) and year.\n";
                    break;
                }
                displayCalendar(month, year);
                break;
            case 2:
                addEvent();
                break;
            case 3:
                viewEvents();
                break;
            case 4:
                deleteEvent();
                break;
            case 5:
                cout << "Exiting\n";
                break;
            default:
                cout << "Invalid choice!\n";
        }
    } while (choice != 5);
    return 0;
}