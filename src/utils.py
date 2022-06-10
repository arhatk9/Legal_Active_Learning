def get_features_from_place_mentions(place):
    feature = []
    if len(place['district']) == 0:
        feature.append(0.0)
    else:
        feature.append(1.0)
    if len(place['state']) == 0:
        feature.append(0.0)
    else:
        feature.append(1.0)
    return feature

def get_features_from_involvingParties_mentions(party):
    feature = []
    feature.append(int(party["appelant"]))
    feature.append(int(party["defendant"]))
    return feature

def get_features_from_citation_mentions(citation):
    feature = []
    if len(citation) == 0:
        feature.append(0.0)
    else:
        feature.append(1.0)
    return feature

def get_features_from_section_mentions(section, chapter):
    feature = []
    if len(section) == 0 or len(chapter):
        feature.append(0.0)
    else:
        feature.append(1.0)
    return feature

def get_features_from_act_mentions(act):
    feature = []
    if len(act) == 0:
        feature.append(0.0)
    else:
        feature.append(1.0)
    return feature

def get_features_from_date_mentions(date):
    feature = []
    if len(date) == 0:
        feature.append(0.0)
    else:
        feature.append(1.0)
    return feature

def get_features_from_sent_length(sentLen):
    feature = []
    if sentLen > self.max_length:
        feature.append(1.0)
    else:
        feature.append(0.0)
    return feature

def get_features_from_relative_position(pos):
    feature = []
    if pos >= 0.0 and pos <= 0.25:
        feature += [0.0, 0.0]
    elif pos > 0.25 and pos <= 0.5:
        feature += [0.0, 1.0]
    elif pos > 0.5 and pos <= 0.75:
        feature += [1.0, 0.0]
    else:
        feature += [1.0, 0.0]
    return feature

def convert_features_to_nos(sents):

    feature_list = []
    label_list = []

    for s in sents:
        f = {}
        f["words"] = [w.lower() for w in word_tokenize(s['sentText'])]
        f["features"] = get_features_from_act_mentions(s["sentReferences"]["acts"]) + \
        get_features_from_citation_mentions(s["sentReferences"]["citations"]) + \
        get_features_from_date_mentions(s["sentEntitites"]["dates"]) + \
        get_features_from_involvingParties_mentions(s["sentEntitites"]["contationsInvolvingParties"]) + \
        get_features_from_place_mentions(s["sentEntitites"]["places"]) + \
        get_features_from_rhetorical_roles(s["sentRhetoricalRole"]) + \
        get_features_from_section_mentions(s['sentReferences']['sections'],s['sentReferences']['chapter']) + \
        [float(s['sentReferences']['statueOrPrecedent'])] + [s["sentForSummary"]] + \
        get_features_from_sent_length(s['sentLength']) + \
        get_features_from_relative_position(s['relativePosition'])

        feature_list.append(f)
        label_list.append(s['sentPseudoRelevance'])
    
    return feature_list, label_list

def read_json_to_list(big_json):

    with open(big_json, 'r') as f:
        all_documents = json.load(f)

    feature_list = []
    label_list = []

    sent = []
    act = []
    citation = []
    date = []
    appealant = []
    defendant = []
    district = []
    state = []
    role = []
    section = []
    precendent = []
    sentLen = []
    relevance = []
    file_name = []

    for key in all_documents:
        if all_documents[key]['judgementSents'] == None:
            continue

        for s in all_documents[key]['judgementSents']:
            f = {}
            f["words"] = [w.lower() for w in word_tokenize(s['sentText'])]
            f["features"] = get_features_from_act_mentions(s["sentReferences"]["acts"]) + \
            get_features_from_citation_mentions(s["sentReferences"]["citations"]) + \
            get_features_from_date_mentions(s["sentEntitites"]["dates"]) + \
            get_features_from_involvingParties_mentions(s["sentEntitites"]["contationsInvolvingParties"]) + \
            get_features_from_place_mentions(s["sentEntitites"]["places"]) + \
            get_features_from_rhetorical_roles(s["sentRhetoricalRole"]) + \
            get_features_from_section_mentions(s['sentReferences']['sections'],s['sentReferences']['chapter']) + \
            [float(s['sentReferences']['statueOrPrecedent'])] + [s["sentForSummary"]] + \
            get_features_from_sent_length(s['sentLength']) + \
            get_features_from_relative_position(s['relativePosition'])
            
            feature_list.append(f)
            label_list.append(s['sentPseudoRelevance'])
            
            x = f["features"]
            # if x[0] != 0.0 or x[1] != 0.0 or x[2] != 0.0 or x[3] != 0.0 or x[4] != 0.0 or x[5] != 0.0 or x[6] != 0.0 or x[-2] != 0.0 or x[-3] != 0.0:
            file_name.append(key)
            sent.append(s['sentText']) 
            act.append(x[0])
            citation.append(x[1])
            date.append(x[2])
            appealant.append(x[3])
            defendant.append(x[4])
            district.append(x[5])
            state.append(x[6])
            role.append(s["sentRhetoricalRole"])
            section.append(x[-3])
            precendent.append(x[-2])
            sentLen.append(s['sentLength'])
            relevance.append(s['sentPseudoRelevance'])
            
    self.csv_data['file_name'] = file_name
    self.csv_data['sent'] = sent
    self.csv_data['act'] = act  #0/1
    self.csv_data['citation'] = citation #0/1
    self.csv_data['date'] = date #0/1
    self.csv_data['appealant'] = appealant #0/1
    self.csv_data['defendant'] = defendant #0/1
    self.csv_data['district'] = district #0/1
    self.csv_data['state'] = state #0/1
    self.csv_data['role'] = role #7 numerals
    self.csv_data['section'] = section #0/1
    self.csv_data['precedent'] = precendent #0/1
    self.csv_data['Length'] = sentLen
    self.csv_data['relevance'] = relevance

    return feature_list, label_list

def get_features_from_rhetorical_roles(self, role):
    return list(le.transform([role])[0])

def create_csv(self):
    pd.DataFrame(self.csv_data).to_csv("./entityfeature.csv")

# le = preprocessing.LabelBinarizer()
# le.fit(["Facts","Argument","Ratio of the decision","Statute","Precedent","Ruling by Present Court","Ruling by Lower Court"])
