# generate processed.csv from raw data

import os
import csv
import datetime

rootDir = '/Users/zhuboyuan/Desktop/school/current/ERG4901/midterm/data'

targetBaseStation = '东莞138工业区F-HLH'


def getTimestampToTraffic():
  timestampToTraffic = {}
  for subDir, _, fileNames in os.walk(rootDir):
    for fileName in fileNames:
      filePath = os.path.join(subDir, fileName)
      if filePath.split('.').pop() == 'csv':
        print(filePath)
        file = open(filePath, encoding='gbk')
        csvReader = csv.reader(file)
        firstRow = True
        for row in csvReader:
          if firstRow:
            firstRow = False
          else:
            timestamp, baseStation, traffic = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M'), row[1], float(row[6])
            if baseStation == targetBaseStation:
              if timestamp in timestampToTraffic:
                timestampToTraffic[timestamp] += traffic
              else:
                timestampToTraffic[timestamp] = traffic
  return timestampToTraffic


if __name__ == '__main__':
  timestampToTraffic = getTimestampToTraffic()
  output = open('./data.csv', 'w')
  writer = csv.writer(output)
  for timestamp in sorted(timestampToTraffic):
    print(timestamp, timestampToTraffic[timestamp])
    writer.writerow([timestamp, timestampToTraffic[timestamp]])
