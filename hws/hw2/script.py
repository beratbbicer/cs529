import numpy as np

num_people, num_events = 18, 14
people, events = [], []
people_events = np.zeros((num_people, num_events)).astype(np.int64)

with open('./assignment2_data/people_by_events_edge_list.csv', 'r') as file:
    file.readline()
    for line in file:
        person, event, weight = line.strip().split(',')

        if person not in people:
            people += [person]

        if event not in events:
            events += [event]

with open('./assignment2_data/people_by_events_edge_list.csv', 'r') as file:
    file.readline()
    for line in file:
        person, event, weight = line.strip().split(',')
        people_events[people.index(person)][events.index(event)] = int(float(weight))

people_people = np.matmul(people_events, people_events.transpose())
event_event = np.matmul(people_events.transpose(), people_events)

with open('./my_data/people_people-graph.csv', 'w') as file:
    file.write('Source,Target,Weight\n')
    for idx in np.ndindex(people_people.shape):
        src, target = people[idx[0]], people[idx[1]]
        file.write(f'{src},{target},{people_people[idx]}\n')

with open('./my_data/event_event-graph.csv', 'w') as file:
    file.write('Source,Target,Weight\n')
    for idx in np.ndindex(event_event.shape):
        src, target = events[idx[0]], events[idx[1]]
        file.write(f'{src},{target},{event_event[idx]}\n')